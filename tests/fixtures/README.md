# Pinned image fixtures (vllm 0.1.dev20051+g487ecf187, image ad0cdd86)

Byte-exact copies of the in-image vLLM sources the kv-offload overlays patch,
captured read-only from the deployed container on 2026-09-01 (Apache-2.0,
SPDX headers retained). The kv-offload patcher tests preflight/apply against
these and exec the patched result under a stubbed vllm namespace, so anchor
drift or a broken port fails on the host before any server is touched.

sha256 (as captured; `shasum -a 256` these files to compare against a future
image):
- offloading/config.py    d400d0b0fadc06f2ad60a1356a6fee730a187dbcc4e48656da523de813419ec9
- offloading/scheduler.py 616e7fd4cb0d09064cbc4d5735f607b37964c6be3b81e26de00d5913e0a9a3e3

Stage-2 captures (2026-09-01, read-only from the live next3 container — same
vllm tree 0.1.dev20051+g487ecf187, verified: the four offloading fixtures'
sha256s match that container byte-exact):
- offloading_connector.py            b5ddf7c1c8c50f6183dcdc4247759865b88e2b1b415e605c7993f615534912e0
- single_type_kv_cache_manager.py    41043976d1d5e38e0465c8004fc04e01f35f66b39a40099c62d79b3756337d00
- core sched/scheduler.py            ec94d3271351b1f4ddb2d3aca5117f8428168855c9a24c2716b73f9987582b7d
  (image_487ecf187_core_sched_scheduler.py — pinned as the RECEIPT for the
  single-group invalid-block path: `(req_block_ids,) =
  self.kv_cache_manager.get_block_ids(req_id)` under the hybrid-allocator
  TODO at lines 2954-2955; it is not a patch target)
