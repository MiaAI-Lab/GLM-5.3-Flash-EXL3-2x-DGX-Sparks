# Building the SM121 image on an air-gapped Spark pair (2026-09-28)

Both target hosts (`spark-d931`, `spark-ea54`) have no route to `registry-1.docker.io`,
`pypi.org` or `codeload.github.com`; only a plain-HTTP Aliyun PyPI mirror answers. This is the
recipe that produced a working image (`5ae1f35d5567`) from recipe `943912c` without rebuilding
anything by hand.

## 1. What the build actually needs from the network

Exactly two third-party artifacts, plus the PyPI mirror:

| dependency | where | why it cannot be fetched on the host |
|---|---|---|
| ExLlamaV3 sources, commit `c5d9c657966ffeeaa9353f0cc899f18629da4a13` | `https://github.com/turboderp-org/exllamav3/archive/<sha>.tar.gz` | `github.com` resolves but the 302 target `codeload.github.com` is unreachable |
| `instanttensor==0.2.0` (cp312, aarch64) | PyPI | `pypi.org` unreachable; mirror only serves HTTP |
| everything else | base image layers + `pip` for the local source tree | already local |

## 2. Vendor both artifacts from a machine that does have internet

```bash
# on the internet-connected machine (any OS):
curl -L -o exllamav3-c5d9c657966ffeeaa9353f0cc899f18629da4a13.tar.gz \
  https://github.com/turboderp-org/exllamav3/archive/c5d9c657966ffeeaa9353f0cc899f18629da4a13.tar.gz
curl -L -o instanttensor-0.2.0-cp312-cp312-manylinux_2_26_aarch64.manylinux_2_28_aarch64.whl \
  https://files.pythonhosted.org/packages/…/instanttensor-0.2.0-cp312-cp312-manylinux_2_26_aarch64.manylinux_2_28_aarch64.whl
sha256sum *.tar.gz *.whl      # record them; mine are 87cf1ed1… and eefde9b1…
scp exllamav3-*.tar.gz instanttensor-*.whl spark-head:~/GLM53-MiaAI-EXL3-recipe/vendor/
```

## 3. Five Dockerfile edits (all local, all reversible)

```dockerfile
# 1. the FROM pin MUST be dropped for an offline build (see §4)
-ARG BASE=vllm/vllm-openai:glm53-flash-arm64-cu130@sha256:905c0293…
+ARG BASE=vllm/vllm-openai:glm53-flash-arm64-cu130

+ENV PIP_INDEX_URL=http://mirrors.aliyun.com/pypi/simple/ \
+    PIP_TRUSTED_HOST=mirrors.aliyun.com \
+    PIP_DISABLE_PIP_VERSION_CHECK=1
+COPY vendor/ /opt/glm53/vendor/

# 2. exllamav3 sources come from the vendored tarball, not codeload
-    curl -fsSL "https://github.com/turboderp-org/exllamav3/archive/${EXLLAMAV3_COMMIT}.tar.gz" \
-      | tar -xz -C /tmp/exllamav3 --strip-components=1; \
+    tar -xz -C /tmp/exllamav3 --strip-components=1 \
+      -f "/opt/glm53/vendor/exllamav3-${EXLLAMAV3_COMMIT}.tar.gz"; \

# 3. instanttensor from the vendored wheel (or the Aliyun mirror)
-RUN pip install --no-deps --no-cache-dir instanttensor==0.2.0 \
+RUN pip install --no-deps --no-cache-dir --find-links /opt/glm53/vendor instanttensor==0.2.0 \
```

Everything else in the build (the CUDA compile of `exllamav3_ext`, the overlay patches, the
in-image tests) is CPU/GPU-only and works offline.

## 4. The BuildKit pitfall that costs the most time

`FROM …@sha256:<digest>` makes BuildKit resolve the manifest **from the registry even when the
image is already local** — the failure is misleading because `docker images` shows the image
present:

```
#2 ERROR: failed to resolve source metadata for docker.io/vllm/vllm-openai:…@sha256:905c0293…
   Head "https://registry-1.docker.io/v2/…/manifests/sha256:905c0293…": dial tcp 74.86.151.167:443: i/o timeout
```

Two checks worth doing when an offline build "cannot find" a local image:

```bash
docker image inspect -f '{{.RepoDigests}}' vllm/vllm-openai:glm53-flash-arm64-cu130
# ["docker.m.daocloud.io/vllm/vllm-openai@sha256:905c0293…"]   <- digest matches, repo name differs
# with the digest dropped, `FROM vllm/vllm-openai:glm53-flash-arm64-cu130` resolves locally:
#   #4 [1/2] FROM docker.io/vllm/vllm-openai:glm53-flash-arm64-cu130
#   #4 CACHED
```

Keep the digest in the file for reproducibility and drop it with `--build-arg BASE=…` (or a
temporary edit) only for the air-gapped build; the local image provably carries the pinned
digest, so nothing shifts.

## 5. Build + ship

```bash
cd ~/GLM53-MiaAI-EXL3-recipe
BUILD=1 SKIP_BUILD=0 SKIP_PULL=1 SKIP_DOWNLOAD=1 SKIP_SHIP=0 SKIP_SYNC=1 ./start.sh start
# .env sets SKIP_BUILD=1 and SKIP_SHIP=1 for this kit, so both must be overridden explicitly.
```

`start.sh` then ships the image to the worker over the CX7 link (`docker save | ssh docker
load`) and runs the in-image GPU self-check before launching.

## 6. Notes

* The recipe-stamp hash covers `vendor/`, so adding or changing a vendored file changes the
  stamp and triggers a rebuild — that is the desired behaviour, but it means `vendor/` should be
  treated as part of the build inputs (and kept out of version control, ~12 MB).
* `SKIP_BUILD=1` in `.env` will *warn* about a stamp mismatch and keep the existing image; that
  is the right mode while iterating on docs or overlay files that are not baked into the image.
* If the host can reach a plain-HTTP PyPI mirror, `pip` needs `PIP_TRUSTED_HOST` as well, which
  is why it is set next to `PIP_INDEX_URL` rather than only the latter.
