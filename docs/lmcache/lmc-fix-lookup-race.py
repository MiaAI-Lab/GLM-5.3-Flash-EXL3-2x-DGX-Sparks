"""Fix LMCache MPConnector defect 5: the async lookup result lost to a race.

Run INSIDE the image container:  python3 lmc-fix-lookup-race.py

THE BUG (measured end-to-end 2026-09-23):
- maybe_submit_lookup_request registers the server-side prefetch job and adds
  the request to _pending_lookups ONLY AFTER every server answered the LOOKUP.
- check_lookup_result treats "request_id not in _pending_lookups" as final and
  returns 0 (adapter ~line 808) — even when the lookup is merely still in
  flight on the MQ.
- update_state_after_alloc (connector ~line 893) calls cleanup_lookup_result
  when the tracker leaves PREFETCHING — vLLM calls it for every scheduled
  request, including ones scheduled with 0 external tokens — discarding the
  pending lookup before the server-side prefetch job completes. The job's
  result (proven: L2->L1 prefetch wrote the chunks) is then unreachable.
- Server-side query_prefetch_status pops the job (exactly-once) and returns 0
  on an empty result — indistinguishable from "the async scan has not run yet"
  when the scan takes longer than the scheduler's first polls.

THE FIX (all ungated — this is the PR fix):
1. Track submission attempts in _submitted_lookups (before the MQ round-trip).
2. check_lookup_result: "not in pending" + was submitted + still healthy and
   no finished result -> return None (keep waiting), not 0.
3. update_state_after_alloc: skip cleanup_lookup_result while the lookup is
   unresolved (has_unresolved_lookup); free the state at request_finished
   instead (added there).
4. Server query_prefetch_status: an empty result within a 2 s grace window
   returns None (job stays pending) instead of consuming the job with 0.
"""
from pathlib import Path
import ast

BASE = Path("/usr/local/lib/python3.12/dist-packages/lmcache/integration/vllm")
SERVER = Path(
    "/usr/local/lib/python3.12/dist-packages/lmcache/v1/multiprocess/modules/lookup.py"
)


def once(text, old, new, tag, expect=1):
    n = text.count(old)
    if n != expect:
        raise RuntimeError(f"{tag}: expected {expect} anchor(s), found {n}")
    return text.replace(old, new)


def patch_file(path, edits, mark):
    text = path.read_text()
    if mark in text:
        print(f"  {path.name}: already patched")
        return
    for old, new, tag, expect in edits:
        text = once(text, old, new, tag, expect)
    path.write_text(text)
    print(f"  {path.name}: patched ({len(edits)} edits)")


if __name__ == "__main__":
    pa = BASE / "vllm_multi_process_adapter.py"
    patch_file(
        pa,
        [
            # A1: track submission attempts
            (
                "        self._lookup_params: dict[str, tuple[list[int], str]] = {}\n",
                "        self._lookup_params: dict[str, tuple[list[int], str]] = {}\n"
                "        # [glm53-lookup-race] Submission attempts (added before the MQ\n"
                "        # round-trip) so check_lookup_result can distinguish \"never\n"
                "        # submitted\" from \"submitted and still unresolved\".\n"
                "        self._submitted_lookups: set[str] = set()\n",
                "a-init",
                1,
            ),
            # A2: record the attempt at submit entry
            (
                "            return\n"
                "\n"
                "        if request_id in self._pending_lookups:\n"
                "            # Skip if there is already a lookup request\n"
                "            return\n",
                "            return\n"
                "\n"
                "        # [glm53-lookup-race] Record the submission attempt even before\n"
                "        # the MQ round-trip completes: a later check_lookup_result must\n"
                "        # not classify an in-flight lookup as \"never submitted\".\n"
                "        self._submitted_lookups.add(request_id)\n"
                "        if request_id in self._pending_lookups:\n"
                "            # Skip if there is already a lookup request\n"
                "            return\n",
                "a-submit",
                1,
            ),
            # A3: keep waiting instead of reporting a premature 0
            (
                "        if request_id not in self._pending_lookups:\n"
                "            # No job — either unhealthy at submit time or already cleaned up.\n"
                "            # Return the cached aggregate if any, otherwise 0.\n"
                "            return self._finished_lookup_results.get(request_id, 0)\n",
                "        if request_id not in self._pending_lookups:\n"
                "            # [glm53-lookup-race] \"Not pending\" has two meanings: the\n"
                "            # lookup was never submitted (unhealthy at submit time —\n"
                "            # report 0), or it was submitted and the add to _pending is\n"
                "            # still in flight (it happens only after every server\n"
                "            # answered the LOOKUP). Reporting 0 for the latter loses the\n"
                "            # hit when the server-side prefetch job completes moments\n"
                "            # later. Keep waiting while healthy.\n"
                "            if (\n"
                "                self.is_healthy\n"
                "                and request_id in self._submitted_lookups\n"
                "                and request_id not in self._finished_lookup_results\n"
                "            ):\n"
                "                return None\n"
                "            return self._finished_lookup_results.get(request_id, 0)\n",
                "a-check",
                1,
            ),
            # A4: unresolved-lookup query used by the connector
            (
                "    def cleanup_lookup_result(self, request_id: str) -> None:\n",
                "    def has_unresolved_lookup(self, request_id: str) -> bool:\n"
                "        \"\"\"[glm53-lookup-race] True while the request's async lookup is\n"
                "        still awaiting its server-side result.\"\"\"\n"
                "        return request_id in self._pending_lookups\n"
                "\n"
                "    def cleanup_lookup_result(self, request_id: str) -> None:\n",
                "a-has-unresolved",
                1,
            ),
            # A5: free the submission marker at cleanup (leak safety)
            (
                "        self._lookup_params.pop(request_id, None)\n",
                "        self._lookup_params.pop(request_id, None)\n"
                "        self._submitted_lookups.discard(request_id)\n",
                "a-cleanup",
                1,
            ),
        ],
        "glm53-lookup-race",
    )

    pc = BASE / "lmcache_mp_connector.py"
    patch_file(
        pc,
        [
            # C1: never discard an unresolved lookup at block allocation
            (
                "            # Clean up lookup future in scheduler adapter\n"
                "            self.scheduler_adapter.cleanup_lookup_result(request.request_id)\n",
                "            # [glm53-lookup-race] Clean up the lookup bookkeeping only when\n"
                "            # the async lookup has resolved. Discarding an unresolved lookup\n"
                "            # here (vLLM allocates blocks for requests scheduled with 0\n"
                "            # external tokens too) makes the next check_lookup_result read\n"
                "            # \"not pending -> 0\" and permanently loses a hit whose\n"
                "            # server-side prefetch job completes later. The state is freed\n"
                "            # at request_finished instead.\n"
                "            if not self.scheduler_adapter.has_unresolved_lookup(\n"
                "                request.request_id\n"
                "            ):\n"
                "                self.scheduler_adapter.cleanup_lookup_result(\n"
                "                    request.request_id\n"
                "                )\n",
                "c-gate-cleanup",
                1,
            ),
            # C2: free the lookup state exactly once at request finish
            (
                "        # Clean up request tracker to prevent memory leak\n"
                "        self._cleanup_request_tracker(request.request_id)\n",
                "        # Clean up request tracker to prevent memory leak\n"
                "        self._cleanup_request_tracker(request.request_id)\n"
                "        # [glm53-lookup-race] Free the lookup bookkeeping at request\n"
                "        # finish; update_state_after_alloc no longer discards unresolved\n"
                "        # lookups, so this is the exactly-once free point.\n"
                "        self.scheduler_adapter.cleanup_lookup_result(request.request_id)\n",
                "c-finish-cleanup",
                1,
            ),
        ],
        "glm53-lookup-race",
    )

    ps = SERVER
    patch_file(
        ps,
        [
            # S1: grace window so an in-flight scan is not consumed as a 0
            (
                "        found = self._ctx.storage_manager.query_prefetch_status(job.handle)\n"
                "        if found is None:\n"
                "            return None\n",
                "        found = self._ctx.storage_manager.query_prefetch_status(job.handle)\n"
                "        if found is None:\n"
                "            return None\n"
                "        # [glm53-lookup-race] An empty result within the grace window is\n"
                "        # indistinguishable from \"the async scan has not produced results\n"
                "        # yet\". Consuming the job here (exactly-once pop) turns a pending\n"
                "        # lookup into a premature 0 and the hit is lost. Hold the job\n"
                "        # pending briefly so the scan can finish.\n"
                "        if not found and (time.monotonic() - job.submit_time) < 2.0:\n"
                "            return None\n",
                "s-grace",
                1,
            ),
        ],
        "glm53-lookup-race",
    )

    for f in (
        BASE / "vllm_multi_process_adapter.py",
        BASE / "lmcache_mp_connector.py",
        SERVER,
    ):
        ast.parse(f.read_text())
    print("lookup-race fix OK (syntax verified on all 3 files)")