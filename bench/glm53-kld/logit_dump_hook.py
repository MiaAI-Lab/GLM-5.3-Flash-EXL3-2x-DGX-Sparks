# SPDX-License-Identifier: Apache-2.0
"""Engine-side taps for GLM-5.3-Flash quant-fidelity captures.

Two independent taps, both inert unless ``GLM_KLD_DUMP_DIR`` is set:

``register_hidden_capture`` - the primary one. Hooks the **output** of the last
stage's ``model.norm`` so the captured tensor is exactly the suite's semantic
point, ``after_final_rmsnorm_before_lm_head``. Read
``<lane>/capture-cut-point.json`` in the published suite before trusting any of
this: those hidden states are already past the final RMSNorm, the shipped
``head/final_norm.safetensors`` exists only so a reader can *verify* the weights,
and replay multiplies by ``head/head.safetensors`` and nothing else. A pre-hook
on this module's input - the obvious thing to write - silently produces a
different quantity, and re-applying the norm on top of the published tensors
rescales every logit row, which inflates KLD while looking perfectly healthy.

The hook is installed through ``LLM.apply_model``, so it lands on every PP stage;
only the last stage owns a real ``RMSNorm`` (``PPMissingLayer`` elsewhere) and
therefore only that stage writes. That is also what makes the capture immune to
MTP spec-decode jitter and to the draft tower's own quantization: nothing here
depends on a sampled token.

Why the module *output* is full-width: vLLM v1 prunes hidden states down to the
logit indices inside the model runner, *after* the model returns. The norm runs
on all prompt tokens, so one 2048-token teacher-forced prefill yields one
2048-row call. Only calls whose row count equals the context length are kept, and
the final row is dropped because it has no next token to predict - which is
exactly the published 2047-row shape. Anything else (decode steps, chunked
prefill fragments, draft passes) is counted and skipped, never guessed at.

``install`` - the secondary tap, for the full-vocabulary logits protocol:
``Sampler.gather_logprobs`` is the last place where the whole (positions, vocab)
matrix exists before vLLM throws all but top-k away, so it is patched to spill
raw logits (use with ``--logprobs-mode raw_logits``).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

ENV_DIR = "GLM_KLD_DUMP_DIR"
ENV_MIN_ROWS = "GLM_KLD_DUMP_MIN_ROWS"
ENV_CONTEXT_ROWS = "GLM_KLD_DUMP_CONTEXT_ROWS"
ENV_HOOKED = "GLM_KLD_DUMP_HOOKED"

#: Tensor key used by the published suite's capture lanes.
TENSOR_KEY = "hidden_states"
#: What the captured tensor is, in the suite's own vocabulary.
SEMANTIC_POINT = "after_final_rmsnorm_before_lm_head"


def _highest(dump_dir: Path, prefix: str, suffix: str) -> int:
    highest = 0
    for path in dump_dir.glob(f"{prefix}-*{suffix}"):
        try:
            highest = max(highest, int(path.stem.split("-")[1]))
        except (IndexError, ValueError):
            continue
    return highest


def _dump(tensor, dump_dir: Path, prefix: str) -> str:
    """Spill one tensor as ``<prefix>-<counter>.safetensors``; return its name."""
    import torch
    from safetensors.torch import save_file

    counter = _highest(dump_dir, prefix, ".safetensors") + 1
    name = f"{prefix}-{counter:06d}.safetensors"
    temporary = dump_dir / (name + ".tmp")
    rows = tensor.detach().to(torch.bfloat16).contiguous().cpu()
    save_file({TENSOR_KEY: rows}, str(temporary))
    os.replace(temporary, dump_dir / name)
    return name


def _hidden_hook(dump_dir: Path, context_rows: int, state: dict):
    def hook(_module, _inputs, output):
        # RMSNorm returns (hidden_states, residual) when a residual is threaded
        # through; the normalized tensor is element 0 either way.
        rows = output[0] if isinstance(output, (tuple, list)) else output
        if not hasattr(rows, "shape") or rows.dim() != 2:
            state["skipped_not_2d"] += 1
            return
        if rows.shape[0] != context_rows:
            # A decode step, a chunked-prefill fragment, or a draft pass. Keeping
            # it would misalign rows against token ids, so it is refused loudly
            # in the aggregate count instead of being quietly resized.
            state["skipped_rows"] += 1
            state["seen_row_counts"].add(int(rows.shape[0]))
            return
        # The last row predicts a token that is not in the context; the suite
        # publishes context_length - 1 rows per context.
        name = _dump(rows[:-1], dump_dir, "hidden")
        state["dumps"] += 1
        state["last"] = name
        state["rows"] = int(rows.shape[0] - 1)
        state["hidden_size"] = int(rows.shape[1])

    return hook


def register_hidden_capture(model) -> str:
    """Install inside an engine worker through ``LLM.apply_model``.

    ``model`` is ``Glm5NextForConditionalGeneration``; the norm lives at
    ``language_model.model.norm`` and is a ``PPMissingLayer`` on every stage but
    the last, which is what makes this a last-stage-only tap.
    """
    directory = os.environ.get(ENV_DIR)
    if not directory:
        return "disabled:GLM_KLD_DUMP_DIR unset"
    dump_dir = Path(directory, "hidden")
    dump_dir.mkdir(parents=True, exist_ok=True)
    context_rows = int(os.environ.get(ENV_CONTEXT_ROWS, "2048"))

    language_model = getattr(model, "language_model", None)
    inner = getattr(language_model, "model", None) if language_model is not None else None
    norm = getattr(inner, "norm", None) if inner is not None else None
    if norm is None:
        raise RuntimeError(
            "no language_model.model.norm on this worker; the model class changed "
            f"or the capture ran on a non-GLM5Next checkpoint: {type(model).__name__}"
        )
    if type(norm).__name__ == "PPMissingLayer":
        return "skipped:PPMissingLayer"

    state: dict = {
        "dumps": 0,
        "skipped_rows": 0,
        "skipped_not_2d": 0,
        "seen_row_counts": set(),
        "rows": 0,
        "hidden_size": 0,
        "last": None,
    }
    model._glm_kld_hidden_state = state
    norm.register_forward_hook(_hidden_hook(dump_dir, context_rows, state))

    weight = getattr(norm, "weight", None)
    hidden_size = int(weight.shape[0]) if weight is not None else 0
    write_json_atomic(dump_dir / "tap.json", {
        "semantic_point": SEMANTIC_POINT,
        "tensor_key": TENSOR_KEY,
        "module": f"language_model.model.norm ({type(norm).__name__})",
        "context_rows_required": context_rows,
        "rows_stored": context_rows - 1,
        "hidden_size": hidden_size,
    })
    return f"hooked:{type(norm).__name__}:hidden={hidden_size}"


# ------------------------------------------------------- logits tap (legacy) ---


def install() -> bool:
    directory = os.environ.get(ENV_DIR)
    if not directory or os.environ.get(ENV_HOOKED) == "1":
        return False
    os.environ[ENV_HOOKED] = "1"

    from vllm.v1.sample.sampler import Sampler

    dump_dir = Path(directory)
    dump_dir.mkdir(parents=True, exist_ok=True)
    min_rows = int(os.environ.get(ENV_MIN_ROWS, "512"))
    counter = {"n": _highest(dump_dir, "logits", ".safetensors")}
    original = (
        Sampler.gather_logprobs.__func__
        if isinstance(Sampler.gather_logprobs, staticmethod)
        else Sampler.gather_logprobs
    )

    def patched(logits, *args, **kwargs):
        # Every TP rank runs this worker; only rank 0 of the last PP stage owns
        # the vocabulary, so dumping is guarded by shape rather than by rank -
        # a rank that does not own the vocab sees a sharded logit matrix whose
        # width is below the vocabulary size.
        try:
            import torch

            rows = int(logits.shape[0])
            width = int(logits.shape[-1])
            if rows >= min_rows and width >= min_rows:
                counter["n"] += 1
                name = f"logits-{counter['n']:06d}.safetensors"
                temporary = dump_dir / (name + ".tmp")
                from safetensors.torch import save_file

                save_file(
                    {"logits": logits.detach().to(torch.float32).contiguous().cpu()},
                    str(temporary),
                )
                os.replace(temporary, dump_dir / name)
                write_json_atomic(dump_dir / f"logits-{counter['n']:06d}.json", {
                    "rows": rows,
                    "vocab": width,
                    "written_at": time.time(),
                })
        except Exception as error:  # never break a serving request over a dump
            write_json_atomic(dump_dir / "error.json", {"error": repr(error)})
        return original(logits, *args, **kwargs)

    Sampler.gather_logprobs = staticmethod(patched)
    print(f"[glm53-kld] logit dump hook installed -> {dump_dir} (min_rows={min_rows})", flush=True)
    return True


def write_json_atomic(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = {key: (sorted(value[key]) if isinstance(value.get(key), set) else value[key])
               for key in value}
    temporary.write_text(json.dumps(payload, sort_keys=True))
    os.replace(temporary, path)


install()


def install_capture(model) -> str:
    """Picklable entry point for ``LLM.apply_model``.

    ``apply_model`` ships this callable to every worker, and pickle resolves
    functions *by reference*: a function defined in the capture script's
    ``__main__`` cannot be resolved inside a spawned worker (whose ``__main__``
    is vLLM's own entry script), which fails with "it's not the same object as
    __main__._install_tap". Living in this module - which every worker already
    imports through PYTHONPATH - is what makes the reference resolvable.
    """
    return register_hidden_capture(model)
