from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import numpy as np
import pytest

import kernel_lab.exl3.backend as backend_module
from kernel_lab.exl3.abi import (
    MCG_MARKER_SIGNED_INT32,
    PackedExl3Metadata,
    PackedExl3Weight,
)
from kernel_lab.exl3.backend import (
    CapabilityReceipt,
    DirectExllamaBackend,
    KernelTactic,
    inspect_capability,
)
from kernel_lab.exl3.benchmark import run_cpu_oracle_check, synthetic_weight
from kernel_lab.exl3.interfaces import KernelKey, Representation, ServingPhase
from kernel_lab.exl3.receipts import TacticMeasurement, build_atlas_receipt, build_catalog
from kernel_lab.exl3.reference import decode_mcg, pack_trellis, unpack_trellis
from kernel_lab.exl3.shapes import GemmWorkload, load_workload_file, parse_shape_spec


def test_metadata_roundtrip_and_exact_k3_geometry() -> None:
    metadata = PackedExl3Metadata(in_features=6144, out_features=2048)

    assert metadata.trellis_shape == (384, 128, 48)
    assert metadata.trellis_payload_bytes == 4_718_592
    assert 3.0 < metadata.effective_bpw < 3.011
    assert PackedExl3Metadata.from_json(metadata.to_json()) == metadata


@pytest.mark.parametrize("bits", range(1, 9))
def test_pack_unpack_roundtrip_for_all_upstream_bitrates(bits: int) -> None:
    rng = np.random.default_rng(bits)
    encoded = rng.integers(0, 1 << bits, size=(2, 3, 256), dtype=np.int16)

    packed = pack_trellis(encoded, bits)
    unpacked = unpack_trellis(packed, bits)

    assert packed.shape == (2, 3, bits * 16)
    assert packed.dtype == np.int16
    assert np.array_equal(unpacked, encoded)


def test_k3_pack_matches_official_cuda_golden_digest() -> None:
    """Golden was measured against ExLlamaV3's CUDA packer on a GB10."""

    encoded = np.arange(2 * 3 * 256, dtype=np.int16).reshape(2, 3, 256) % 8
    packed = pack_trellis(encoded, 3)

    assert hashlib.sha256(packed.tobytes()).hexdigest() == (
        "e543cefbeede83f0d0b015ad8bd1b85918f8f7d74966f557c9065490f36d15e5"
    )


def test_mcg_decode_matches_official_cuda_golden_values() -> None:
    expected = np.array(
        [
            1.84375,
            0.134521484375,
            -0.75927734375,
            0.83984375,
            0.6650390625,
            -0.460693359375,
            -1.66015625,
            -1.1103515625,
            0.123046875,
            0.4638671875,
            -1.703125,
            -0.6083984375,
            -0.184814453125,
            0.9404296875,
            0.1552734375,
            -0.623046875,
        ],
        dtype=np.float16,
    )

    actual = decode_mcg(np.arange(16, dtype=np.int16))

    assert np.array_equal(actual, expected)


def test_weight_validation_fails_closed_on_wrong_marker() -> None:
    metadata = PackedExl3Metadata(128, 128)
    with pytest.raises(ValueError, match="marker"):
        PackedExl3Weight(
            metadata,
            np.zeros(metadata.trellis_shape, dtype=np.int16),
            np.ones(128, dtype=np.float16),
            np.ones(128, dtype=np.float16),
            np.array([0], dtype=np.int32),
        )


def test_streaming_matmul_matches_materialized_oracle() -> None:
    result = run_cpu_oracle_check(seed=17)

    assert result["passed"]
    assert result["max_abs_error"] < 3e-6


def test_unknown_future_model_shape_requires_no_registry() -> None:
    workload = parse_shape_spec(
        "novel_moe.branch_17:7168:2304",
        model_id="publisher/model-released-after-kernel-lab",
        tp_world_size=4,
        phase="vision_decode_v2",
    )

    assert workload == GemmWorkload(
        operator="novel_moe.branch_17",
        k=7168,
        n=2304,
        model_id="publisher/model-released-after-kernel-lab",
        tp_world_size=4,
        phase="vision_decode_v2",
    )


def test_arbitrary_workload_file(tmp_path) -> None:
    path = tmp_path / "workloads.json"
    path.write_text(
        '{"workloads":[{"operator":"future.linear","k":5120,"n":1280,'
        '"model_id":"future/model","tp_world_size":8,"phase":"decode"}]}',
        encoding="utf-8",
    )

    assert load_workload_file(path) == (
        GemmWorkload("future.linear", 5120, 1280, "future/model", 8),
    )


def test_synthetic_weight_has_no_dense_payload() -> None:
    weight = synthetic_weight(128, 128, seed=3, pack_codes=True)

    assert weight.tensor_manifest() == {
        "trellis": {"dtype": "int16", "shape": [8, 8, 48], "nbytes": 6144},
        "suh": {"dtype": "float16", "shape": [128], "nbytes": 256},
        "svh": {"dtype": "float16", "shape": [128], "nbytes": 256},
        "mcg": {"dtype": "int32", "shape": [1], "nbytes": 4},
    }
    assert int(weight.mcg[0]) == MCG_MARKER_SIGNED_INT32


def test_future_dispatch_key_is_explicit_without_implementing_later_phases() -> None:
    key = KernelKey(
        compute_capability=(12, 1),
        phase=ServingPhase.DECODE,
        representation=Representation.exl3_mcg(3),
        m=1,
        n=2048,
        k=6144,
    )

    assert not key.grouped
    assert key.representation.abi_name == "exllamav3.exl3.mcg"
    assert isinstance(inspect_capability().reason, str)


def test_metadata_accepts_every_upstream_integer_bitrate() -> None:
    for bits in range(1, 9):
        metadata = PackedExl3Metadata(128, 128, bits=bits)
        assert metadata.bits == bits
        assert metadata.trellis_shape == (8, 8, 16 * bits)


def test_direct_backend_has_no_model_or_operator_allowlist(monkeypatch) -> None:
    capability = CapabilityReceipt(
        available=True,
        reason="test",
        cuda_version="13.0",
        compute_capability=(12, 1),
        device_name="NVIDIA GB10",
        driver_version="580.00",
        multiprocessor_count=20,
    )
    monkeypatch.setattr(backend_module, "inspect_capability", lambda: capability)
    key = KernelKey(
        compute_capability=(12, 1),
        phase="decode",
        representation=Representation.exl3_mcg(7),
        m=128,
        n=2304,
        k=7168,
        model_id="publisher/model-from-the-future",
        operator="unseen.operator.branch",
        tp_world_size=16,
    )

    assert DirectExllamaBackend().supports(key) == (True, "supported")


def test_grouped_workload_fails_closed_before_grouped_kernel_phase(monkeypatch) -> None:
    capability = CapabilityReceipt(
        available=True,
        reason="test",
        cuda_version="13.0",
        compute_capability=(12, 1),
        device_name="NVIDIA GB10",
        driver_version="580.00",
        multiprocessor_count=20,
    )
    monkeypatch.setattr(backend_module, "inspect_capability", lambda: capability)
    key = KernelKey(
        (12, 1),
        "decode",
        Representation.exl3_mcg(3),
        1,
        2048,
        4096,
        grouped=True,
    )

    supported, reason = DirectExllamaBackend().supports(key)
    assert not supported
    assert "grouped MoE" in reason


def test_canonical_atlas_receipt_is_model_agnostic() -> None:
    generated_at = datetime.now(timezone.utc)
    capability = CapabilityReceipt(
        available=True,
        reason="test",
        torch_version="2.test",
        cuda_version="13.0",
        compute_capability=(12, 1),
        device_name="NVIDIA GB10",
        device_uuid="GPU-test",
        driver_version="580.00",
        multiprocessor_count=20,
        extension_path="/test/exllamav3_ext.so",
    )
    workload = GemmWorkload(
        "future.operator", 7168, 2304, "publisher/future-model", 4
    )
    receipt = build_atlas_receipt(
        generated_at=generated_at,
        capability=capability,
        metadata=PackedExl3Metadata(7168, 2304, bits=3),
        workload=workload,
        m=2,
        tactic=KernelTactic("shape2-sms20", 2, 20),
        measurement=TacticMeasurement(
            latency_ms=0.1,
            effective_bandwidth_gbps=500.0,
            achieved_tflops=1.0,
            reconstruction_overhead_pct=70.0,
            max_abs_error=0.001,
            mean_abs_error=0.0001,
            bottleneck="memory_bandwidth_or_pipeline",
            selected_shape=2,
            passed=True,
        ),
        runtime_commit="a" * 40,
        command=["python", "-m", "kernel_lab.exl3.benchmark"],
        seed=0,
    )

    assert receipt["schema_version"] == "atlas.kernel-benchmark/v1"
    assert receipt["workload"]["model_id"] == "publisher/future-model"
    assert receipt["workload"]["operator"] == "future.operator"
    assert receipt["backend"]["kernel_name"] == "exl3_gemm.shape2.sms20"
    assert receipt["representation"]["full_precision_materialized"] is False
    assert build_catalog([receipt])["receipts"] == [receipt]


def test_receipt_names_current_upstream_small_m_path() -> None:
    receipt = build_atlas_receipt(
        generated_at=datetime.now(timezone.utc),
        capability=CapabilityReceipt(
            True,
            "test",
            cuda_version="13.0",
            compute_capability=(12, 1),
            device_name="NVIDIA GB10",
            driver_version="580.00",
            multiprocessor_count=20,
        ),
        metadata=PackedExl3Metadata(4096, 4096, bits=3),
        workload=GemmWorkload("linear", 4096, 4096),
        m=1,
        tactic=KernelTactic("auto", -1, 0),
        measurement=TacticMeasurement(
            0.1,
            400.0,
            1.0,
            50.0,
            0.001,
            0.0001,
            "memory_bandwidth_or_pipeline",
            90,
            True,
        ),
        runtime_commit="b" * 40,
        command=["benchmark"],
        seed=0,
    )

    assert receipt["backend"]["kernel_name"] == "exl3_gemv.qtip_small_m"
