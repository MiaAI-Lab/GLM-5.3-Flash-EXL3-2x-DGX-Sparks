#!/usr/bin/env python3
"""Adaptive verification length for DFlash2 (env-gated, default OFF).

The drafter keeps its trained 8-token block (1 anchor + 7 drafts). This patch
lets the scheduler verify only a per-step *prefix* of the 7 draft tokens, chosen
from a CPU-side EMA of accepted draft tokens per request, and keeps the batch
uniform (FLASHINFER_MLA_SPARSE_SM120 and the KDA backend only support
uniform-batch FULL graphs), so it applies the minimum over all running spec
requests. It also captures one uniform decode FULL graph per candidate length.

Knobs (all read at runtime inside the container):
  GLM53_ADAPTIVE_K            off (default) | ema
  GLM53_ADAPTIVE_K_SET        candidate draft lengths, default "2,4,7"
  GLM53_ADAPTIVE_K_ALPHA      EMA alpha, default 0.25
  GLM53_ADAPTIVE_K_MARGIN     n = largest set value <= ceil(ema + margin), default 1.0
  GLM53_ADAPTIVE_K_MIN_STEPS  steps observed at full length before trimming, default 4
  GLM53_ADAPTIVE_K_SATURATE   observation fed to the EMA when every verified row
                              was accepted: "max" (default: the full draft length,
                              so the EMA can climb back above the current n) or
                              "n" (the accepted count itself; ratchets downward)
  GLM53_ADAPTIVE_K_HIST       print the chosen-length histogram every N steps, 200
  GLM53_ADAPTIVE_K_FILE       runtime override JSON (default /root/.cache/vllm/glm53_adaptive_k.json,
                              i.e. <CACHE_ROOT>/glm53_adaptive_k.json on the head): keys mode, alpha,
                              margin, set (clamped to the boot-time set), saturate, min_steps.
                              Re-read when its mtime changes (checked every 50 steps). Only honoured
                              when the knob was on at boot (graphs exist for the boot-time lengths).

Structured-output requests and requests with fewer than MIN_STEPS observed
verify steps stay at the full length. Requests padded with -1 placeholders on
their first decode step (C>1) are not distinguished; MIN_STEPS covers that.

Patches (idempotent, fail closed on drifted anchors, marker comment):
  vllm/v1/core/sched/scheduler.py           observe (update_from_output); choose the per-step draft
                                            slot count in schedule() (num_spec_tokens_to_schedule,
                                            which the AsyncScheduler uses to size its placeholders —
                                            the path this kit runs); trim in update_draft_token_ids
                                            for the synchronous path
  vllm/v1/worker/gpu/cudagraph_utils.py     extra uniform decode graph lengths
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

SITE = Path("/usr/local/lib/python3.12/dist-packages/vllm")
SCHED = Path(os.environ.get("GLM53_SCHEDULER_PY", SITE / "v1/core/sched/scheduler.py"))
CG = Path(os.environ.get("GLM53_CUDAGRAPH_UTILS_PY", SITE / "v1/worker/gpu/cudagraph_utils.py"))
MARK = "# [glm53-adaptive-k]"

IMPORT_OLD = "import itertools\nimport time\n"
IMPORT_NEW = "import itertools\nimport os\nimport time\n"

SCHED_HELPER = '''
class _Glm53AdaptiveK:  # [glm53-adaptive-k]
    """CPU-only EMA policy for the verified draft prefix length."""

    def __init__(self) -> None:
        mode = os.environ.get("GLM53_ADAPTIVE_K", "off").strip().lower()
        self.enabled = mode in ("ema", "on", "1")
        self.alpha = float(os.environ.get("GLM53_ADAPTIVE_K_ALPHA", "0.25"))
        self.margin = float(os.environ.get("GLM53_ADAPTIVE_K_MARGIN", "1.0"))
        self.min_steps = int(os.environ.get("GLM53_ADAPTIVE_K_MIN_STEPS", "4"))
        raw = os.environ.get("GLM53_ADAPTIVE_K_SET", "2,4,7")
        self.k_set = sorted({int(x) for x in raw.split(",") if x.strip()})
        self.saturate = os.environ.get("GLM53_ADAPTIVE_K_SATURATE", "max").strip().lower()
        self.hist_every = int(os.environ.get("GLM53_ADAPTIVE_K_HIST", "200"))
        self.state: dict[str, list[float]] = {}  # req_id -> [ema, observed_steps]
        self.hist: dict[int, int] = {}
        self.steps = 0
        self.k_max = max(self.k_set) if self.k_set else 0
        # Runtime override (no reboot): JSON {"mode","alpha","margin","set","saturate","min_steps"}
        # at GLM53_ADAPTIVE_K_FILE (default: the mounted vLLM cache dir). "set" is clamped to
        # the boot-time set because graphs are captured for the boot-time lengths only.
        self.boot_set = list(self.k_set)
        self.boot_enabled = self.enabled
        self.file = os.environ.get("GLM53_ADAPTIVE_K_FILE", "/root/.cache/vllm/glm53_adaptive_k.json")
        self.file_mtime = None
        self._reload()
        if self.enabled:
            print(
                f"[glm53-adaptive-k] enabled set={self.k_set} alpha={self.alpha} "
                f"margin={self.margin} min_steps={self.min_steps} saturate={self.saturate}",
                flush=True,
            )

    def _reload(self) -> None:
        if not self.boot_enabled:
            return  # graphs for the extra lengths exist only when enabled at boot
        try:
            mtime = os.stat(self.file).st_mtime
        except OSError:
            mtime = None
        if mtime == self.file_mtime:
            return
        self.file_mtime = mtime
        if mtime is None:
            return
        try:
            import json
            with open(self.file) as fh:
                cfg = json.load(fh)
            mode = str(cfg.get("mode", "ema")).strip().lower()
            self.enabled = mode in ("ema", "on", "1")
            self.alpha = float(cfg.get("alpha", self.alpha))
            self.margin = float(cfg.get("margin", self.margin))
            self.min_steps = int(cfg.get("min_steps", self.min_steps))
            self.saturate = str(cfg.get("saturate", self.saturate)).strip().lower()
            if "set" in cfg:
                want = {int(x) for x in (cfg["set"] if isinstance(cfg["set"], list) else str(cfg["set"]).split(","))}
                self.k_set = sorted(want & set(self.boot_set)) or list(self.boot_set)
            self.state.clear()
            self.hist.clear()
            print(
                f"[glm53-adaptive-k] reloaded {self.file}: enabled={self.enabled} set={self.k_set} "
                f"alpha={self.alpha} margin={self.margin} min_steps={self.min_steps} saturate={self.saturate}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[glm53-adaptive-k] override file ignored: {exc!r}", flush=True)

    def observe(self, req_id: str, num_draft: int, num_accepted: int) -> None:
        if not self.enabled or num_draft <= 0:
            return
        if num_accepted >= num_draft and self.saturate != "n":
            obs = float(max(self.k_max, num_draft))
        else:
            obs = float(num_accepted)
        st = self.state.get(req_id)
        if st is None:
            self.state[req_id] = [obs * self.alpha + float(self.k_max) * (1.0 - self.alpha), 1.0]
        else:
            st[0] = obs * self.alpha + st[0] * (1.0 - self.alpha)
            st[1] += 1.0

    def choose(self, req_id: str, k: int, structured: bool):
        """Draft length for one request, or None when it must stay at full length."""
        if not self.enabled or structured or k <= 0:
            return None
        st = self.state.get(req_id)
        if st is None or st[1] < self.min_steps:
            return None
        import math
        target = int(math.ceil(st[0] + self.margin))
        cands = [v for v in self.k_set if v <= min(target, k)]
        n = max(cands) if cands else min(self.k_set)
        return max(1, min(n, k))

    def apply(self, reqs, live_ids) -> None:
        """reqs: list of (request, structured). Trims spec_token_ids to a uniform n.

        A structured-output or not-yet-observed request pins the whole batch at
        the full length (uniform batch, nothing trimmed)."""
        if self.steps % 50 == 0:
            self._reload()
        if not self.enabled or not reqs:
            self.steps += 1
            return
        ns = []
        for r, s in reqs:
            n_i = self.choose(r.request_id, len(r.spec_token_ids), s)
            if n_i is None:
                ns = None
                break
            ns.append(n_i)
        n = max(len(r.spec_token_ids) for r, _ in reqs) if ns is None else min(ns)
        for r, _ in reqs:
            if len(r.spec_token_ids) > n:
                r.spec_token_ids = r.spec_token_ids[:n]
        self._count(n, live_ids)

    def batch_k(self, k: int, reqs, live_ids) -> int:
        """Schedule-time hook (async scheduler): the number of draft slots every
        request gets on the next step. Minimum over the scheduled decode
        requests; any structured-output or not-yet-observed request pins the
        batch at k."""
        if self.steps % 50 == 0:
            self._reload()
        if not self.enabled or k <= 0:
            self.steps += 1
            return k
        ns = []
        for r in reqs:
            if r is None or getattr(r, "is_prefill_chunk", False):
                continue
            n_i = self.choose(r.request_id, k, bool(getattr(r, "use_structured_output", False)))
            if n_i is None:
                ns = None
                break
            ns.append(n_i)
        n = k if not ns else min(ns)
        self._count(n, live_ids)
        return n

    def _count(self, n: int, live_ids) -> None:
        self.hist[n] = self.hist.get(n, 0) + 1
        self.steps += 1
        if self.hist_every > 0 and self.steps % self.hist_every == 0:
            total = sum(self.hist.values())
            parts = " ".join(f"{k}:{v}" for k, v in sorted(self.hist.items()))
            emas = " ".join(f"{rid[:8]}={st[0]:.2f}/{int(st[1])}" for rid, st in list(self.state.items())[:4])
            print(f"[glm53-adaptive-k] step {self.steps} chosen-length hist ({total}): {parts} | ema {emas}", flush=True)
            self.state = {rid: st for rid, st in self.state.items() if rid in live_ids}


_GLM53_ADAPTIVE_K = _Glm53AdaptiveK()  # [glm53-adaptive-k]


'''

OBS_OLD = """                num_draft_tokens = len(scheduled_spec_token_ids)
                num_sampled = self.num_sampled_tokens_per_step
                num_accepted = max(len(generated_token_ids) - num_sampled, 0)
"""
OBS_NEW = """                num_draft_tokens = len(scheduled_spec_token_ids)
                num_sampled = self.num_sampled_tokens_per_step
                num_accepted = max(len(generated_token_ids) - num_sampled, 0)
                _GLM53_ADAPTIVE_K.observe(req_id, num_draft_tokens, num_accepted)  # [glm53-adaptive-k]
"""

UPD_OLD = """    def update_draft_token_ids(self, draft_token_ids: DraftTokenIds) -> None:
        for req_id, spec_token_ids in zip(
            draft_token_ids.req_ids,
            draft_token_ids.draft_token_ids,
        ):
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # The request may have been finished. Skip.
                continue

            if request.is_prefill_chunk:
                # Ignore draft tokens for prefill chunks.
                if request.spec_token_ids:
                    request.spec_token_ids = []
                continue

            # Add newly generated spec token ids to the request.
            if self.structured_output_manager.should_advance(request):
                metadata = request.structured_output_request
                spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)  # type: ignore[union-attr]
            request.spec_token_ids = spec_token_ids
"""
UPD_NEW = """    def update_draft_token_ids(self, draft_token_ids: DraftTokenIds) -> None:
        _ak_reqs = []  # [glm53-adaptive-k]
        for req_id, spec_token_ids in zip(
            draft_token_ids.req_ids,
            draft_token_ids.draft_token_ids,
        ):
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # The request may have been finished. Skip.
                continue

            if request.is_prefill_chunk:
                # Ignore draft tokens for prefill chunks.
                if request.spec_token_ids:
                    request.spec_token_ids = []
                continue

            # Add newly generated spec token ids to the request.
            _ak_structured = self.structured_output_manager.should_advance(request)  # [glm53-adaptive-k]
            if _ak_structured:
                metadata = request.structured_output_request
                spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)  # type: ignore[union-attr]
            request.spec_token_ids = spec_token_ids
            if _GLM53_ADAPTIVE_K.boot_enabled and spec_token_ids:  # [glm53-adaptive-k]
                _ak_reqs.append((request, _ak_structured))
        if _ak_reqs:  # [glm53-adaptive-k]
            _GLM53_ADAPTIVE_K.apply(_ak_reqs, self.requests)
"""

SCHED_K_OLD = """        # Dynamic speculative decoding: compute optimal K
        num_spec_tokens_to_schedule = self.num_spec_tokens
        if self.dynamic_sd_lookup is not None and len(num_scheduled_tokens) > 0:
            num_spec_tokens_to_schedule = self.dynamic_sd_lookup[
                len(num_scheduled_tokens)
            ]
"""
SCHED_K_NEW = """        # Dynamic speculative decoding: compute optimal K
        num_spec_tokens_to_schedule = self.num_spec_tokens
        if self.dynamic_sd_lookup is not None and len(num_scheduled_tokens) > 0:
            num_spec_tokens_to_schedule = self.dynamic_sd_lookup[
                len(num_scheduled_tokens)
            ]
        if _GLM53_ADAPTIVE_K.boot_enabled and self.dynamic_sd_lookup is None and num_scheduled_tokens:  # [glm53-adaptive-k]
            num_spec_tokens_to_schedule = _GLM53_ADAPTIVE_K.batch_k(
                num_spec_tokens_to_schedule,
                [self.requests.get(_rid) for _rid in num_scheduled_tokens],
                self.requests,
            )
"""

CG_HELPER = '''
def _glm53_adaptive_k_query_lens(lens, decode_query_len):  # [glm53-adaptive-k]
    """Extra uniform decode graph lengths for the adaptive verification prefix."""
    import os

    mode = os.environ.get("GLM53_ADAPTIVE_K", "off").strip().lower()
    if mode not in ("ema", "on", "1"):
        return lens
    raw = os.environ.get("GLM53_ADAPTIVE_K_SET", "2,4,7")
    ks = {int(x) for x in raw.split(",") if x.strip()}
    extra = {k + 1 for k in ks if 0 < k + 1 <= decode_query_len}
    out = sorted(set(lens) | extra | {decode_query_len})
    print(f"[glm53-adaptive-k] uniform decode graph query lens: {out}", flush=True)
    return out


'''
CG_ANCHOR = "@dataclass(frozen=True)\nclass BatchExecutionDescriptor:\n"
CG_OLD = """        else:
            decode_query_lens = [self.decode_query_len]
"""
CG_NEW = """        else:
            decode_query_lens = [self.decode_query_len]
        decode_query_lens = _glm53_adaptive_k_query_lens(decode_query_lens, self.decode_query_len)  # [glm53-adaptive-k]
"""


def replace_once(path: Path, text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{path}: expected one {label} target, found {n}")
    return text.replace(old, new, 1)


def patch_scheduler() -> None:
    if not SCHED.is_file():
        raise SystemExit(f"missing {SCHED}")
    text = SCHED.read_text()
    if MARK in text:
        print(f"{SCHED.name}: {MARK} already present — skipping")
        return
    if "import os\n" not in text.split("import time\n", 1)[0]:
        text = replace_once(SCHED, text, IMPORT_OLD, IMPORT_NEW, "import os")
    needle = "from vllm.compilation.cuda_graph import CUDAGraphStat\n"
    if text.count(needle) != 1:
        raise SystemExit(f"{SCHED}: helper insert point not unique")
    text = text.replace(needle, SCHED_HELPER + needle, 1)
    text = replace_once(SCHED, text, OBS_OLD, OBS_NEW, "observe")
    text = replace_once(SCHED, text, UPD_OLD, UPD_NEW, "update_draft_token_ids")
    text = replace_once(SCHED, text, SCHED_K_OLD, SCHED_K_NEW, "num_spec_tokens_to_schedule")
    SCHED.write_text(text)
    print(f"patched {SCHED.name} (GLM53_ADAPTIVE_K={os.environ.get('GLM53_ADAPTIVE_K', 'off')})")


def patch_cudagraph_utils() -> None:
    if not CG.is_file():
        raise SystemExit(f"missing {CG}")
    text = CG.read_text()
    if MARK in text:
        print(f"{CG.name}: {MARK} already present — skipping")
        return
    if text.count(CG_ANCHOR) != 1:
        raise SystemExit(f"{CG}: helper insert point not unique")
    text = text.replace(CG_ANCHOR, CG_HELPER + CG_ANCHOR, 1)
    text = replace_once(CG, text, CG_OLD, CG_NEW, "decode_query_lens")
    CG.write_text(text)
    print(f"patched {CG.name} (adaptive-k graph lengths)")


def main() -> int:
    patch_scheduler()
    patch_cudagraph_utils()
    return 0


if __name__ == "__main__":
    sys.exit(main())
