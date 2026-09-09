"""Strict adapter and tactic surface for ExLlamaV3's direct EXL3 kernel."""

from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass
from typing import Any

from .abi import PackedExl3Weight
from .interfaces import KernelKey, ServingPhase


@dataclass(frozen=True)
class CapabilityReceipt:
    available: bool
    reason: str
    torch_version: str | None = None
    cuda_version: str | None = None
    compute_capability: tuple[int, int] | None = None
    device_name: str | None = None
    device_uuid: str | None = None
    driver_version: str | None = None
    multiprocessor_count: int | None = None
    extension_path: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def inspect_capability() -> CapabilityReceipt:
    try:
        import torch
    except ImportError:
        return CapabilityReceipt(False, "PyTorch is not installed")
    if not torch.cuda.is_available():
        return CapabilityReceipt(
            False,
            "CUDA is not available",
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda,
        )
    capability = tuple(int(v) for v in torch.cuda.get_device_capability())
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    device_name = str(properties.name)
    device_uuid = str(getattr(properties, "uuid", "")) or None
    multiprocessor_count = int(properties.multi_processor_count)
    driver_version = _driver_version()
    if capability != (12, 1):
        return CapabilityReceipt(
            False,
            f"Milestone 0 requires SM121, found sm_{capability[0]}{capability[1]}",
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda,
            compute_capability=capability,
            device_name=device_name,
            device_uuid=device_uuid,
            driver_version=driver_version,
            multiprocessor_count=multiprocessor_count,
        )
    try:
        import exllamav3_ext
    except ImportError as exc:
        return CapabilityReceipt(
            False,
            f"exllamav3_ext is unavailable: {exc}",
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda,
            compute_capability=capability,
            device_name=device_name,
            device_uuid=device_uuid,
            driver_version=driver_version,
            multiprocessor_count=multiprocessor_count,
        )
    return CapabilityReceipt(
        True,
        "native SM121 EXL3 extension is available",
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        compute_capability=capability,
        device_name=device_name,
        device_uuid=device_uuid,
        driver_version=driver_version,
        multiprocessor_count=multiprocessor_count,
        extension_path=getattr(exllamav3_ext, "__file__", None),
    )


def _driver_version() -> str | None:
    """Read the installed driver identity without guessing from CUDA runtime."""

    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    value = completed.stdout.splitlines()[0].strip() if completed.stdout else ""
    return value or None


@dataclass(frozen=True)
class KernelTactic:
    """One directly measurable ExLlamaV3 launch configuration."""

    name: str
    force_shape_idx: int
    force_num_sms: int

    @property
    def kernel_name(self) -> str:
        if self.force_shape_idx <= 0:
            return "exl3_gemm.autotuned"
        return f"exl3_gemm.shape{self.force_shape_idx}.sms{self.force_num_sms}"


class DirectExllamaBackend:
    """Single-matrix direct path; grouped MoE and prefill are out of scope."""

    name = "exllamav3_direct_exl3"
    max_m = 144

    def supports(self, key: KernelKey) -> tuple[bool, str]:
        receipt = inspect_capability()
        if not receipt.available:
            return False, receipt.reason
        if key.phase != ServingPhase.DECODE:
            return False, "the first direct-kernel sweep implements decode only"
        representation = key.representation
        if (
            representation.format != "exl3"
            or representation.abi_name != "exllamav3.exl3.mcg"
            or representation.abi_version != 1
            or representation.codebook != "mcg"
            or not representation.bits_per_weight.is_integer()
            or not 1 <= int(representation.bits_per_weight) <= 8
        ):
            return False, "the portable direct adapter requires EXL3 MCG ABI v1 at integer K1-K8"
        if key.grouped:
            return False, "this tactic sweep measures one matrix; grouped MoE is a later phase"
        if key.m < 1 or key.m > self.max_m:
            return False, f"the pinned direct kernel requires 1 <= M <= {self.max_m}"
        return True, "supported"

    @staticmethod
    def _linear_class():
        try:
            from overlay.exl3 import load_linear_exl3_cls

            return load_linear_exl3_cls()
        except ImportError:
            from exllamav3.modules.quant.exl3 import LinearEXL3

            return LinearEXL3

    @staticmethod
    def _extension():
        try:
            from overlay.exl3 import load_exllamav3_ext

            return load_exllamav3_ext()
        except ImportError:
            import exllamav3_ext

            return exllamav3_ext

    def build(self, weight: PackedExl3Weight):
        receipt = inspect_capability()
        if not receipt.available:
            raise RuntimeError(receipt.reason)
        import torch

        weight.validate()
        device = torch.device("cuda")
        linear_class = self._linear_class()
        return linear_class(
            config=None,
            in_features=weight.metadata.in_features,
            out_features=weight.metadata.out_features,
            trellis=torch.from_numpy(weight.trellis).to(device=device),
            suh=torch.from_numpy(weight.suh).to(device=device),
            svh=torch.from_numpy(weight.svh).to(device=device),
            mcg=torch.from_numpy(weight.mcg.reshape(1)).to(device=device),
            out_dtype=torch.float16,
            transformers_fix=True,
        )

    def run(self, x, linear):
        """Run the direct path and fail if ExLlama attempts full reconstruction."""

        import torch

        if x.ndim != 2 or not (1 <= x.shape[0] <= self.max_m):
            raise ValueError(f"x must be [M,K] with 1 <= M <= {self.max_m}")

        def reconstruction_is_forbidden(*_args, **_kwargs):
            raise RuntimeError("direct EXL3 backend attempted a full weight reconstruction")

        original = linear.reconstruct_hgemm
        linear.reconstruct_hgemm = reconstruction_is_forbidden
        try:
            return linear.forward(x.contiguous().half(), {"reconstruct": False}, out_dtype=torch.float32)
        finally:
            linear.reconstruct_hgemm = original

    def tactics(self, key: KernelKey) -> tuple[KernelTactic, ...]:
        """Enumerate compatible tile shapes and conservative SM quotas."""

        supported, reason = self.supports(key)
        if not supported:
            raise RuntimeError(reason)
        extension = self._extension()
        receipt = inspect_capability()
        if receipt.multiprocessor_count is None:
            raise RuntimeError("SM count is unavailable")
        sm_total = receipt.multiprocessor_count
        sm_counts = sorted({max(1, sm_total // 4), max(1, sm_total // 2), sm_total})
        bits = int(key.representation.bits_per_weight)
        tactics = [KernelTactic("auto", -1, 0)]
        for shape_idx in range(1, int(extension.exl3_gemm_num_kernel_shapes()) + 1):
            if not extension.exl3_gemm_shape_compat(
                shape_idx, key.m, key.k, key.n, bits
            ):
                continue
            tactics.extend(
                KernelTactic(f"shape{shape_idx}-sms{sms}", shape_idx, sms)
                for sms in sm_counts
            )
        return tuple(tactics)

    def run_tactic(
        self,
        x,
        linear,
        tactic: KernelTactic,
        *,
        output=None,
        x_hadamard=None,
    ):
        """Run one direct launch; no reconstruction fallback is reachable."""

        import torch

        if x.ndim != 2 or not (1 <= x.shape[0] <= self.max_m):
            raise ValueError(f"x must be [M,K] with 1 <= M <= {self.max_m}")
        extension = self._extension()
        if output is None:
            output = torch.empty(
                (x.shape[0], linear.out_features),
                dtype=torch.float32,
                device=x.device,
            )
        if x_hadamard is None:
            x_hadamard = torch.empty_like(x)
        selected_shape = extension.exl3_gemm(
            x.contiguous().half(),
            linear.trellis,
            output,
            linear.suh,
            x_hadamard,
            linear.svh,
            tactic.force_shape_idx,
            linear.mcg,
            linear.mul1,
            tactic.force_num_sms,
        )
        return output, int(selected_shape)
