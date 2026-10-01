#!/usr/bin/env python3
"""CPU-only TP4 performance configuration and runtime wiring regression tests."""
import ast
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
KNOBS = ('GLM53_DENSE_FP8', 'GLM53_KDA_BF16_LARGE_M', 'GLM53_EXL3_MOE_FAST')


def launch(caller=None, shared='', topology='', probe='main restart', docker_rc=0, worker_rc=0, missing_rank=-1):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        s = (ROOT / 'start-tp4.sh').read_text().rsplit('main "$@"', 1)[0]
        s += '''
validate_loadclone_artifacts() { :; }
banner() { :; }
start() { echo HOST_START; }
stop() { echo HOST_STOP; }
status() { echo HOST_STATUS; }
logs() { echo HOST_LOGS; }
docker() { if [ "$1" = image ]; then [ "$MISSING_RANK" != 0 ]; return; fi; echo "DOCKER $*" >&2; return ''' + str(docker_rc) + '''; }
worker_ssh_n() { if [[ "$2" == "docker image inspect"* ]]; then [ "$MISSING_RANK" != "$1" ]; return; fi; echo "WORKER_PROBE $1 $2" >&2; return ''' + str(worker_rc) + '''; }
ssh() { echo UNEXPECTED_SSH >&2; return 99; }
scp() { echo UNEXPECTED_SCP >&2; return 99; }
python3() { echo UNEXPECTED_PYTHON >&2; return 99; }
''' + probe + '\n'
        (root / 'start-tp4.sh').write_text(s)
        (root / '.env').write_text(shared)
        (root / '.env.tp4').write_text(topology)
        (root / 'overlay').symlink_to(ROOT / 'overlay', target_is_directory=True)
        return subprocess.run(['bash', str(root / 'start-tp4.sh')], text=True, capture_output=True,
                              env={'PATH': os.environ['PATH'], 'HOME': directory, 'USER': 'test', 'MISSING_RANK': str(missing_rank), **(caller or {})})


def test_tp4_does_not_inherit_shared_performance():
    r = launch(shared='GLM53_DENSE_FP8=all\nGLM53_KDA_BF16_LARGE_M=1\nGLM53_EXL3_MOE_FAST=1\n',
               probe='printf "%s|%s|%s" "$GLM53_DENSE_FP8" "$GLM53_KDA_BF16_LARGE_M" "$GLM53_EXL3_MOE_FAST"')
    assert r.returncode == 0 and r.stdout == 'off|0|0', (r.returncode, r.stdout, r.stderr)


def test_caller_setness():
    dotenv = ''.join(f'{k}=file\n' for k in KNOBS)
    for value in ('caller', ''):
        r = launch(dict.fromkeys(KNOBS, value), dotenv, dotenv,
                   'printf "[%s]\\n" ' + ' '.join(f'"${k}"' for k in KNOBS))
        assert r.returncode == 0 and r.stdout.splitlines() == [f'[{value}]'] * 3, r


def test_invalid_before_actions_and_kda_dependency():
    cases = [({k: value}, k) for k in KNOBS for value in ('bogus', '2', '', 'on', ' 1', 'ALL', 'dense, kda')]
    cases += [({'GLM53_DENSE_FP8': v}, KNOBS[0]) for v in ('0', '1', 'no', 'none', ',,', 'kda,', 'all,kda')]
    cases += [({'GLM53_KDA_BF16_LARGE_M': '1'}, 'requires kda'),
              ({'GLM53_DENSE_FP8': 'all', 'GLM53_DENSE_EXL3': '1'}, 'GLM53_DENSE_EXL3')]
    for caller, error in cases:
        for cmd in ('start', 'restart'):
            r = launch(caller, probe=f'main {cmd}')
            assert r.returncode == 2 and error in r.stderr, (caller, r.stderr)
            assert 'HOST_' not in r.stdout and 'DOCKER' not in r.stderr and 'UNEXPECTED_' not in r.stderr


def test_management_and_off_skip_probes():
    for cmd in ('stop', 'status', 'logs'):
        r = launch(dict.fromkeys(KNOBS, 'invalid'), probe=f'main {cmd}')
        assert r.returncode == 0 and r.stdout.strip() == 'HOST_' + cmd.upper(), r
        assert 'DOCKER' not in r.stderr and 'UNEXPECTED_' not in r.stderr
    r = launch({'GLM53_EXL3_MOE_FAST': '1', 'MAX_NUM_SEQS': 'invalid'})
    assert r.returncode == 2 and 'DOCKER' not in r.stderr and 'WORKER_PROBE' not in r.stderr, r
    r = launch(docker_rc=99)
    assert r.returncode == 0 and r.stdout.splitlines() == ['HOST_STOP', 'HOST_START'], r
    assert 'DOCKER' not in r.stderr and 'UNEXPECTED_' not in r.stderr


def test_performance_sources_and_acceptance():
    values = dict(zip(KNOBS, ('all', '1', '1')))
    for caller, topology, origin in ((values, ''.join(f'{k}=invalid\n' for k in KNOBS), 'caller environment'),
                                    ({}, ''.join(f'{k}={v}\n' for k, v in values.items()), '.env.tp4')):
        r = launch(caller, topology=topology, docker_rc=1)
        assert r.returncode == 2 and 'BUILD=1' in r.stderr and 'HOST_' not in r.stdout, r
        assert 'SKIP_PULL=1' in r.stderr, r
        r = launch(caller, topology=topology)
        assert r.returncode == 0 and 'glm53_fast_moe_version() == 1' in r.stderr, r
        assert r.stdout.count('tp4 perf:') == 1
        for name, value in zip(('dense_fp8', 'kda_bf16', 'fast'), values.values()):
            assert f'{name}={value} ({origin})' in r.stdout, r
    for groups in ('off', 'shared', 'dense', 'mla', 'kda', 'dense,kda', 'shared,dense,kda,mla', 'all', 'kda,kda'):
        r = launch({'GLM53_DENSE_FP8': groups, 'GLM53_KDA_BF16_LARGE_M': str(int('kda' in groups or groups == 'all'))})
        assert r.returncode == 0 and 'DOCKER' not in r.stderr, r


def test_missing_image_skips_only_pre_stop_probe():
    for rank in range(4):
        r = launch({'GLM53_EXL3_MOE_FAST': '1'}, missing_rank=rank,
                   docker_rc=int(rank == 0), worker_rc=0)
        assert r.returncode == 0 and 'HOST_STOP' in r.stdout, r
        assert ('DOCKER run' in r.stderr) == (rank != 0), r
        for worker in (1, 2, 3):
            assert (f'WORKER_PROBE {worker} ' in r.stderr) == (rank != worker), r
        r = launch({'GLM53_EXL3_MOE_FAST': '1'}, missing_rank=rank,
                   probe='validate_tp4_performance_artifacts')
        assert r.returncode == 0 and 'DOCKER run' in r.stderr and r.stderr.count('WORKER_PROBE') == 3, r
    r = launch({'GLM53_EXL3_MOE_FAST': '1'}, missing_rank=0, docker_rc=1,
               probe='validate_tp4_performance_artifacts')
    assert r.returncode == 2 and 'BUILD=1' in r.stderr, r


def test_missing_artifacts_and_worker_probe_fail_before_stop():
    for knob in ({'GLM53_DENSE_FP8': 'dense'}, {'GLM53_EXL3_MOE_FAST': '1'}):
        r = launch(knob, probe='TP4_EXL3_OVERLAY_HOST=/nonexistent; main restart')
        assert r.returncode == 2 and 'overlay missing' in r.stderr and 'HOST_' not in r.stdout, r
    r = launch(probe='TP4_EXL3_OVERLAY_HOST=/nonexistent; TP4_DENSE_FP8_PATCH_HOST=/nonexistent; main restart')
    assert r.returncode == 0, r
    r = launch({'GLM53_EXL3_MOE_FAST': '1'}, worker_rc=1)
    assert r.returncode == 2 and 'rank 1' in r.stderr and 'BUILD=1' in r.stderr and 'HOST_' not in r.stdout, r


def test_wiring_mounts_delivery_order_and_off_trace():
    s = (ROOT / 'start-tp4.sh').read_text()
    bodies = re.findall(r"<<'EOF'\n(.*?)\nEOF", s, re.S)
    assert len(bodies) == 2
    expected = [f'/opt/glm53/patch_{name}.py' for name in (
        'glm_video_placeholders', 'suppress_stops_in_reasoning', 'scheduler_decode_floor',
        'mamba_align_chunking', 'glm5_drafter_group', 'hybrid_prefix_hit',
        'apc_per_group_retention', 'mamba_align_state_free', 'xgrammar_termination',
        'kpool_tail_slotmap', 'kpool_tail_seed_stride', 'spinwait', 'loadclone',
        'sparse_mla_slice', 'indexer_workspace', 'ablit')]
    for body in bodies:
        subprocess.run(['bash', '-n'], input=body, text=True, check=True)
        for enabled in (False, True):
            code = body[body.index('# The checkout runtime'):body.index('if [ "${ABLIT:-0}"')]
            code = code.replace('[ -f /opt/glm53/patch_dense_fp8.py ]', str(enabled).lower())
            code = re.sub(r'\[ -f /opt/glm53/[^ ]+ \]', 'true', code)
            r = subprocess.run(['bash', '-c', 'python3() { echo "$*"; };\n' + code],
                               env={}, capture_output=True, text=True)
            assert r.returncode == 0, r.stderr
            assert r.stdout.splitlines() == (['/opt/glm53/patch_dense_fp8.py'] if enabled else []) + expected
    mount_block = s[s.index('    local -a perf_mounts=()'):s.index('    docker rm -f', s.index('launch_cluster()'))]
    for dense, fast, count in (('off', '0', 0), ('dense', '0', 4), ('off', '1', 4)):
        code = 'tp4_performance_enabled() { [ "$GLM53_DENSE_FP8" != off ] || [ "$GLM53_EXL3_MOE_FAST" = 1 ]; }; f() {\n'
        code += mount_block + '\nprintf "%s|%s" "${#perf_mounts[@]}" "$worker_perf_mounts"; }; f'
        r = subprocess.run(['bash', '-c', code], capture_output=True, text=True,
                           env={'GLM53_DENSE_FP8': dense, 'GLM53_EXL3_MOE_FAST': fast,
                                'TP4_EXL3_OVERLAY_HOST': '/checkout/exl3.py', 'TP4_DENSE_FP8_PATCH_HOST': '/checkout/patch_dense_fp8.py'})
        assert r.returncode == 0 and r.stdout.startswith(str(count) + '|'), r
        assert ('/tmp/glm53-exl3.py' in r.stdout) == bool(count), r
    for k, value in zip(KNOBS, ('off', '0', '0')):
        assert f'\n{k}={value}\n' in (ROOT / '.env.tp4.example').read_text()


def test_tp4_shape_contract():
    tree = ast.parse((ROOT / 'overlay/exl3.py').read_text())
    shapes = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == 'KDA_BF16_LARGE_M_SHAPES_BY_TP' for t in n.targets))
    # q/k/v/b shard across heads; f_a/g_a remain replicated (128 each).
    derived = {tp: ((((64 + tp - 1) // tp) * 385 + 256), 4096) for tp in (2, 3, 4)}
    assert shapes == derived
    from test_dense_fp8_patch import _load_helpers
    ns = _load_helpers({'_GLM53_TP3_UNALIGNED_KDA_SUFFIXES', '_glm53_use_marlin'})
    for suffix in ('f_b_proj', 'g_b_proj', 'in_proj_qkvbfg_a'):
        assert ns['_glm53_use_marlin']('kda', 'model.layers.0.self_attn.' + suffix, 4)


if __name__ == '__main__':
    failed = 0
    for name, fn in sorted(list(globals().items())):
        if name.startswith('test_'):
            try:
                fn()
                print('ok', name)
            except Exception as exc:
                failed += 1
                print('FAIL', name, repr(exc))
    sys.exit(bool(failed))
