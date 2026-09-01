"""Container-side probe for the KLD harness (no GPU allocation)."""
import inspect
import os
import traceback

import logit_dump_hook


def main() -> int:
    print("hook module imported at interpreter start:", os.environ.get("GLM_KLD_DUMP_HOOKED"))
    if not os.environ.get("GLM_KLD_DUMP_HOOKED"):
        try:
            logit_dump_hook.install()
            print("manual install ok")
        except Exception:
            traceback.print_exc()
    from vllm.v1.sample.sampler import Sampler

    print("gather_logprobs now:", getattr(Sampler.gather_logprobs, "__qualname__", None))
    from vllm import LLM
    from vllm.engine.arg_utils import EngineArgs

    named = set(inspect.signature(LLM.__init__).parameters)
    engine_fields = {field.name for field in __import__("dataclasses").fields(EngineArgs)}
    accepted = named | engine_fields
    needed = {
        "model", "dtype", "tensor_parallel_size", "pipeline_parallel_size",
        "enable_expert_parallel", "kv_cache_dtype", "gpu_memory_utilization",
        "max_model_len", "max_num_batched_tokens", "max_num_seqs", "enforce_eager",
        "seed", "disable_log_stats", "skip_mm_profiling", "limit_mm_per_prompt",
        "enable_prefix_caching", "attention_config", "logprobs_mode",
    }
    print("kwargs missing from LLM/EngineArgs:", sorted(needed - accepted))
    print("LLM.apply_model:", callable(getattr(LLM, "apply_model", None)))
    from vllm.inputs import TokensPrompt  # noqa: F401
    from vllm.sampling_params import RequestOutputKind  # noqa: F401
    import numpy
    import safetensors
    import torch

    print("deps:", torch.__version__, numpy.__version__, safetensors.__version__)
    print("register_hidden_capture callable:", callable(logit_dump_hook.register_hidden_capture))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
