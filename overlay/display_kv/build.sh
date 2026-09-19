#!/usr/bin/env bash
# Build libglm53_display_kv.so inside the serving image (same glibc/CUDA as the
# worker) and drop it next to this script. Skipped when the .so is newer than
# display_kv.c. Usage: build.sh [IMAGE]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${1:-${IMAGE:-glm53-flash-sm121:nvme}}"
SRC="$HERE/display_kv.c"
OUT="$HERE/libglm53_display_kv.so"
if [ -f "$OUT" ] && [ "$OUT" -nt "$SRC" ] && [ "$OUT" -nt "${BASH_SOURCE[0]}" ]; then
    exit 0
fi
echo "[display-kv] building $(basename "$OUT") in $IMAGE" >&2
docker run --rm --user "$(id -u):$(id -g)" -v "$HERE:/w" -w /w --entrypoint bash "$IMAGE" -c \
    'gcc -O2 -fPIC -shared -Wall -I/usr/local/cuda/include -o /tmp/lib.so display_kv.c -L/usr/local/cuda/lib64/stubs -lcuda \
     && cp /tmp/lib.so libglm53_display_kv.so.tmp'
mv -f "$OUT.tmp" "$OUT"
chmod 755 "$OUT"
echo "[display-kv] built $OUT" >&2
