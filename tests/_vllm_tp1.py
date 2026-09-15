#!/usr/bin/env python3
"""Real single-rank vLLM model-parallel lifecycle for standalone harnesses.

Standalone entry points (no engine) leave vLLM's distributed state at its
initial ``_TP=None``, so any ``process_weights_after_loading()`` path that
calls ``get_tensor_model_parallel_world_size()`` fails with::

    AssertionError: tensor model parallel group is not initialized

This module initializes a REAL single-rank (TP=1) environment using the
actual vLLM APIs of the pinned revision — a genuine process group plus
genuine ``GroupCoordinator`` objects — and destroys it exception-safely
afterward. No mocks, no monkeypatching, no fake world-size variables.

Usage::

    from _vllm_tp1 import single_rank_model_parallel

    with single_rank_model_parallel():  # backend="nccl" on CUDA, else "gloo"
        meth.process_weights_after_loading(layer)
"""

from __future__ import annotations

import contextlib
import os
import socket
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _free_loopback_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])
    finally:
        s.close()


@contextlib.contextmanager
def single_rank_model_parallel(backend: str | None = None):
    """Initialize real TP=1 vLLM distributed state; destroy it on exit.

    Entry asserts no model-parallel state leaks in from an earlier test;
    exit always destroys (even on exception) and asserts the process is
    clean again so initialized state can never leak between tests.
    """
    import torch
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        get_tensor_model_parallel_world_size,
        init_distributed_environment,
        initialize_model_parallel,
        model_parallel_is_initialized,
    )

    if model_parallel_is_initialized() or torch.distributed.is_initialized():
        raise RuntimeError(
            "single_rank_model_parallel: distributed state already "
            "initialized on entry (leaked from an earlier test?)")
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method=(
            f"tcp://127.0.0.1:{_free_loopback_port()}"),
        local_rank=0,
        backend=backend,
    )
    try:
        # initialize_model_parallel reads parallel sizes from the current
        # vLLM config; a default-constructed VllmConfig is the real config
        # object with TP=1/PP=1/DP=1 (model_config stays None — nothing on
        # this path reads it).
        with set_current_vllm_config(VllmConfig()):
            initialize_model_parallel(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
            )
        world_size = get_tensor_model_parallel_world_size()
        assert world_size == 1, f"expected TP=1, got {world_size}"
        yield world_size
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()
        assert not torch.distributed.is_initialized(), \
            "torch process group leaked after destroy"
        assert not model_parallel_is_initialized(), \
            "vLLM model-parallel state leaked after destroy"
