# SPDX-License-Identifier: AGPL-3.0-only
# Adapted from carlosduque-incoxe's DeepSeek PR #19 (8d2cdf9), which credits
# @Saolence PR #3 for ring integration and FujitsuPolycom/sparkring for NCCL.
# No network configuration changes; sourced only by the TP4 launcher.

ring_validate() {
    case ${NCCL_SWITCHLESS_RING_ONLY:-0} in 0|1) ;; *) return 1 ;; esac
    [[ ${NCCL_SWITCHLESS_RING_ONLY:-0} == 1 ]] || return 0
    [[ $TP == 4 && $NNODES == 4 && $USE_HOST_NCCL == 1 ]] || {
        echo 'Ring requires TP=NNODES=4 and USE_HOST_NCCL=1' >&2; return 1;
    }
    [[ ${NCCL_IB_SUBNET_PREFIX_LEN:-24} =~ ^([1-9]|[12][0-9]|3[0-2])$ ]] || return 1
    [[ ${NCCL_NCHANNELS:-4} =~ ^[1-9][0-9]{0,2}$ ]] || return 1
    local -A seen=()
    local address
    for address in "$HEAD_IP" "$WORKER_IP" "$WORKER2_IP" "$WORKER3_IP"; do
        [[ -n $address && $address != *CHANGE_ME* && ! ${seen[$address]+yes} ]] || {
            echo 'Set four distinct reachable management IPs in .env.tp4' >&2; return 1;
        }
        seen[$address]=1
    done
}

# Validate BOTH cable-facing HCAs, not just the first entry. GID index may
# differ between ranks, but must be valid on both selected ports per rank.
ring_check_gid() {
    local root=${1:-/sys/class/infiniband} entry base hca port gid
    local -a ports
    IFS=, read -r -a ports <<<"${RING_HCAS#=}"
    [[ ${#ports[@]} == 2 && ${ports[0]} != "${ports[1]}" ]] || {
        echo 'Ring requires two distinct exact HCAs (optionally :port)' >&2; return 1;
    }
    [[ $RING_GID =~ ^[0-9]+$ ]] || return 1
    local -A seen_ports=()
    for entry in "${ports[@]}"; do
        [[ $entry =~ ^[a-zA-Z0-9_]+(:[1-9][0-9]*)?$ ]] || return 1
        hca=${entry%%:*}; port=1
        [[ $entry != *:* ]] || port=${entry##*:}
        [[ ! ${seen_ports[$hca:$port]+yes} ]] || return 1
        seen_ports[$hca:$port]=1
        base=$root/$hca/ports/$port
        [[ $(cat "$base/state") == '4: ACTIVE' ]] || return 1
        [[ $(cat "$base/gid_attrs/types/$RING_GID") == 'RoCE v2' ]] || return 1
        gid=$(cat "$base/gids/$RING_GID")
        [[ $gid =~ ^0000:0000:0000:0000:0000:ffff:[[:xdigit:]]{4}:[[:xdigit:]]{4}$ && $gid != *:0000:0000 ]] || return 1
    done
}

# This function is sent over SSH with quoted assignments, not eval'ed input.
ring_probe_node() {
    set -euo pipefail
    ring_check_gid
    local lib=$RING_DIR/$NCCL_SO_NAME image_id path hash
    [[ $lib == /* && $lib != *:* && -r $lib ]] || return 1
    grep -qa SWITCHLESS_RING_ONLY "$lib" || {
        echo "Missing sparkring marker in $lib" >&2; return 1;
    }
    hash=$(sha256sum "$lib"); hash=${hash%% *}
    image_id=$(docker image inspect --format '{{.Id}}' "$IMAGE")
    # CPU-only, no GPUs, no network, no image pull, read-only root filesystem.
    path=$(docker run --rm --pull=never --network none --read-only --cap-drop ALL \
        --entrypoint python3 "$IMAGE" -c '
import pathlib, sys
paths = {str((pathlib.Path(p)/"nvidia/nccl/lib/libnccl.so.2").absolute())
         for p in sys.path if (pathlib.Path(p)/"nvidia/nccl/lib/libnccl.so.2").is_file()}
assert len(paths) == 1, "Expected exactly one pip NCCL library: %r" % paths
print(paths.pop())')
    [[ $path == /* && $path != *:* && $path != *$'\n'* ]] || return 1
    if [[ $RING_REQUIRE_IDLE == 1 ]]; then
        local processes
        processes=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
        [[ -z $processes ]] || {
            echo 'GPU workload present; stop it explicitly before GLM launch' >&2; return 1;
        }
    fi
    printf '%s\t%s\t%s\n' "$image_id" "$hash" "$path"
}

ring_preflight_all() {
    [[ ${NCCL_SWITCHLESS_RING_ONLY:-0} == 1 ]] || return 0
    ring_validate || return 1
    local r result baseline='' script RING_DIR RING_HCAS RING_GID
    local RING_REQUIRE_IDLE=${1:-0}
    RING_MOUNT_PATHS=()
    for r in 0 1 2 3; do
        if [[ $r == 0 ]]; then
            RING_DIR=$NCCL_HOST_DIR; RING_HCAS=$HEAD_CX7_IB; RING_GID=$HEAD_GID
        else
            RING_DIR=$(_tp4_rank_nccl_dir "$r")
            RING_HCAS=$(_tp4_rank_cx7_ib "$r"); RING_GID=$(_tp4_rank_gid "$r")
        fi
        script=$(printf 'set -euo pipefail\n';
            printf '%s=%q\n' RING_DIR "$RING_DIR" RING_HCAS "$RING_HCAS" \
                RING_GID "$RING_GID" NCCL_SO_NAME "$NCCL_SO_NAME" IMAGE "$IMAGE" \
                RING_REQUIRE_IDLE "$RING_REQUIRE_IDLE";
            declare -f ring_check_gid ring_probe_node; printf '\nring_probe_node\n')
        if [[ $r == 0 ]]; then
            result=$(bash -s <<<"$script") || { echo "Ring preflight failed on rank $r" >&2; return 1; }
        else
            result=$(worker_ssh_n "$r" 'bash -s' <<<"$script") || {
                echo "Ring preflight failed on rank $r" >&2; return 1;
            }
        fi
        [[ -n $baseline ]] || baseline=$result
        [[ $result == "$baseline" ]] || {
            echo "Rank $r image / NCCL SHA256 / overlay path differs from rank 0" >&2; return 1;
        }
        RING_MOUNT_PATHS[$r]=${result##*$'\t'}
        echo "Ring rank $r: GID, patched NCCL and image identity OK" >&2
    done
}

ring_env_args() {
    [[ ${NCCL_SWITCHLESS_RING_ONLY:-0} == 1 ]] || return 0
    local -n args=$1
    args+=(-e NCCL_SWITCHLESS_RING_ONLY=1 -e NCCL_ALGO=Ring
        -e NCCL_SKIP_TREE_CONNECT=1 -e NCCL_IB_SUBNET_AWARE_ROUTING=1
        -e "NCCL_IB_SUBNET_PREFIX_LEN=${NCCL_IB_SUBNET_PREFIX_LEN:-24}"
        -e NCCL_P2P_LEVEL=SYS)
    if [[ -z ${NCCL_NCHANNELS:-} ]]; then
        args+=(-e NCCL_MIN_NCHANNELS=4 -e NCCL_MAX_NCHANNELS=4)
    fi
}
