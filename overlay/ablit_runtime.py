# SPDX-License-Identifier: MIT
"""GLM-5.3-Flash o_proj abliteration (ABLIT) — load-time refusal-direction edit.

Applies the classic weight orthogonalization to every ``self_attn.o_proj`` in
the configured layer range:

    W' = (I - alpha * r r^T / ||r||^2) W

``r`` is the refusal direction in the residual/output space of o_proj
(hidden_size=4096). Components of the attention output orthogonal to ``r``
are preserved exactly; the component along ``r`` is scaled by ``1 - alpha``
(alpha > 1 over-projects, which is what the published recipe uses).

METHOD (ABLIT_METHOD=auto | transplant | proj):
  transplant — replace the stock o_proj weights L15-45 (incl. the checkpoint
    MTP block's, layer 45) with the donor tensors fetched by
    ablit/fetch_transplant.py into ablit/transplant/ (raw bf16 + MANIFEST.json).
    This is the published "dealign-oproj-transplant" edit — a byte-copy from
    dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4 into the stock body. Direction
    orthogonalization does NOT work on this model (the published directions
    are statistically random vs stock o_proj — measured), so transplant is
    the only effective mode; "auto" prefers it when the tensors are present.
  proj — the classic orthogonalization below. Kept for custom directions.

TP notes: o_proj is a RowParallelLinear — vLLM shards the *input* dim, the
4096 output rows are replicated on every rank. The edit only touches rows,
so each rank applies the identical formula to its own shard and the
post-allreduce result stays consistent. No collective is needed.

The edit runs at the end of ``Glm5NextModel.load_weights`` / ``Glm5NextMTP.
load_weights`` (installed by overlay/patch_ablit.py). o_proj stays native
BF16 in this serve (attn is unquantized), so the parameter is final at that
point — quantized-expert post-processing never touches it, and CUDA graph
capture happens later.

Artifacts (shipped in ``ablit/`` and mounted at /opt/glm53/ablit):
  LAYER_MAP.json                              layer/shard map + published recipe
  refusal_direction_glm53_dealign_late.pt     published dealign direction
  refusal_direction_glm53_bf_oproj.pt         blackfrost direction (alpha_ref 3.0)

Source: drowzeys/keys-GLM-5.3-Flash-NVFP4-ablit-l15-45-anchorstock
(published method "dealign-oproj-transplant": layers 15-45 edited, 0-14 kept
as stock safety anchors, MTP block included).

Env knobs (see .env.example):
  ABLIT=1                  enable (default off — hook is a no-op)
  ABLIT_DIR                artifact dir (default /opt/glm53/ablit)
  ABLIT_METHOD             auto | transplant | proj (default auto: transplant
                           when ablit/transplant/ is populated, else proj)
  ABLIT_DIRECTION          dealign | bf_oproj | /path/to/dir.pt (default dealign)
  ABLIT_LAYERS             inclusive ranges, e.g. "15-45" or "15,17-19"
  ABLIT_ALPHA              projection scale (default 3.0)
  ABLIT_INCLUDE_MTP        also edit the checkpoint MTP block when it exists
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any

import torch

def _tp_world() -> int:
    try:
        from vllm.distributed import get_tensor_model_parallel_world_size
        return get_tensor_model_parallel_world_size()
    except Exception:  # standalone / tests
        return 1


def _tp_rank() -> int:
    try:
        from vllm.distributed import get_tensor_model_parallel_rank
        return get_tensor_model_parallel_rank()
    except Exception:  # standalone / tests
        return 0


try:  # inside the vLLM image
    from vllm.logger import init_logger

    logger = init_logger(__name__)
except Exception:  # standalone (tests)
    import logging

    logger = logging.getLogger("glm53_ablit")

DEFAULT_ABLIT_DIR = "/opt/glm53/ablit"
TRANSPLANT_SUBDIR = "transplant"

DIRECTION_FILES = {
    "dealign": "refusal_direction_glm53_dealign_late.pt",
    "bf_oproj": "refusal_direction_glm53_bf_oproj.pt",
}

# paths inside the hooked model that carry an o_proj to edit
_LAYER_O_PROJ_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.self_attn\.o_proj$")
_MTP_O_PROJ_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.mtp_block\.self_attn\.o_proj$")


class AblitError(RuntimeError):
    """ABLIT was explicitly enabled but the edit cannot be applied."""


def env_flag(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None or val == "":
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def parse_layers(spec: str) -> list[int]:
    """Parse "15-45" / "15,17-19" into a sorted list of layer indices."""
    layers: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            try:
                lo, hi = int(lo_s), int(hi_s)
            except ValueError as exc:
                raise AblitError(f"ABLIT_LAYERS: bad range {part!r}") from exc
            if hi < lo:
                raise AblitError(f"ABLIT_LAYERS: inverted range {part!r}")
            layers.update(range(lo, hi + 1))
        else:
            try:
                layers.add(int(part))
            except ValueError as exc:
                raise AblitError(f"ABLIT_LAYERS: bad index {part!r}") from exc
    if not layers:
        raise AblitError("ABLIT_LAYERS resolved to an empty set")
    return sorted(layers)


def resolve_direction_path(ablit_dir: str, direction: str) -> Path:
    if "/" in direction:  # explicit path to a .pt
        return Path(direction)
    fname = DIRECTION_FILES.get(direction)
    if fname is None:
        raise AblitError(
            f"ABLIT_DIRECTION={direction!r} not one of "
            f"{sorted(DIRECTION_FILES)} (or a .pt path)"
        )
    return Path(ablit_dir) / fname


def load_direction(path: Path) -> torch.Tensor:
    try:
        obj = torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise AblitError(f"cannot load ablit direction {path}: {exc}") from exc
    if not isinstance(obj, dict) or "directions" not in obj:
        raise AblitError(f"{path}: expected a dict with 'directions'")
    r = obj["directions"]
    if not torch.is_tensor(r):
        raise AblitError(f"{path}: 'directions' must be a tensor, got {type(r)}")
    if r.dim() == 2 and r.shape[0] == 1:  # some exports store a [1, N] row
        r = r.squeeze(0)
    if r.dim() != 1:
        raise AblitError(f"{path}: 'directions' must be 1-D (or [1, N]), got {tuple(r.shape)}")
    r = r.to(torch.float32)
    norm = r.norm()
    if not torch.isfinite(norm) or float(norm) <= 0:
        raise AblitError(f"{path}: direction has non-positive/invalid norm {norm}")
    return r


def apply_to_o_proj(mod: Any, r: torch.Tensor, alpha: float) -> dict[str, Any]:
    """Orthogonalize one o_proj module in place. Returns a report dict."""
    weight = getattr(mod, "weight", None)
    if weight is None or not torch.is_tensor(weight) or weight.dim() != 2:
        return {"edited": False, "reason": "no 2-D .weight"}
    if weight.shape[0] != r.numel():
        return {
            "edited": False,
            "reason": f"out_features={weight.shape[0]} != direction dim {r.numel()}",
        }
    r_dev = r.to(device=weight.device, dtype=torch.float32)
    with torch.no_grad():
        w32 = weight.data.to(torch.float32)
        # W' = W - alpha * outer(r, r @ W) / ||r||^2
        rw = r_dev @ w32  # [in_local]
        delta = alpha * torch.outer(r_dev, rw) / float(r_dev @ r_dev)
        w32.sub_(delta)
        weight.data.copy_(w32.to(weight.dtype))
        residual = float((r_dev @ weight.data.to(torch.float32)).abs().max())
    return {
        "edited": True,
        "shape": tuple(weight.shape),
        "residual_max": residual,
    }


def walk_o_proj(model: Any) -> list[tuple[str, int | None, Any]]:
    """Collect (name, layer_idx or None-if-pure-mtp, module) o_proj candidates."""
    found: list[tuple[str, int | None, Any]] = []
    seen: set[int] = set()
    for name, mod in model.named_modules():
        if id(mod) in seen:
            continue
        m = _LAYER_O_PROJ_RE.search(name)
        if m is not None:
            seen.add(id(mod))
            found.append((name, int(m.group(1)), mod))
            continue
        m = _MTP_O_PROJ_RE.search(name)
        if m is not None:
            seen.add(id(mod))
            found.append((name, int(m.group(1)), mod))
    return found


def unwrap_text_model(model: Any) -> Any:
    """Accept the multimodal wrapper and hand back the text model."""
    lm = getattr(model, "language_model", None)
    if lm is not None:
        inner = getattr(lm, "model", None)
        if inner is not None:
            return inner
        return lm
    return model


def apply_ablit(
    model: Any,
    r: torch.Tensor,
    layers: list[int],
    alpha: float,
    include_mtp: bool,
) -> dict[str, Any]:
    """Edit every configured o_proj under ``model``. Fails hard on surprises."""
    text_model = unwrap_text_model(model)
    candidates = walk_o_proj(text_model)
    want = set(layers)
    report: dict[str, Any] = {
        "edited_layers": [],
        "skipped": [],
        "mtp_edited": False,
    }
    seen_ids: set[int] = set()
    for name, idx, mod in candidates:
        if id(mod) in seen_ids:
            continue
        seen_ids.add(id(mod))
        is_mtp = _MTP_O_PROJ_RE.search(name) is not None
        if is_mtp:
            if not include_mtp:
                report["skipped"].append({"name": name, "reason": "include_mtp=0"})
                continue
            if idx not in want:
                report["skipped"].append(
                    {"name": name, "reason": f"layer {idx} not in ABLIT_LAYERS"}
                )
                continue
        else:
            if idx not in want:
                continue
        rep = apply_to_o_proj(mod, r, alpha)
        if rep.get("edited"):
            if is_mtp:
                report["mtp_edited"] = True
            else:
                report["edited_layers"].append(idx)
            logger.info(
                "ablit: orthogonalized %s shape=%s residual_max=%.3e",
                name,
                rep.get("shape"),
                rep.get("residual_max", float("nan")),
            )
        else:
            report["skipped"].append({"name": name, "reason": rep.get("reason")})
            logger.warning("ablit: skipped %s: %s", name, rep.get("reason"))
    report["edited_layers"].sort()
    return report


def load_transplant_tensors(
    ablit_dir: str, layers: list[int]
) -> dict[int, torch.Tensor]:
    """Load donor o_proj tensors (full, unsharded) from ablit/transplant/."""
    tdir = Path(ablit_dir) / TRANSPLANT_SUBDIR
    manifest_path = tdir / "MANIFEST.json"
    if not manifest_path.is_file():
        raise AblitError(f"ABLIT_METHOD=transplant but {manifest_path} is missing "
                         "— run ablit/fetch_transplant.py first")
    manifest = json.loads(manifest_path.read_text())
    meta = {int(k): v for k, v in (manifest.get("layers") or manifest.get("tensors") or {}).items()}
    out: dict[int, torch.Tensor] = {}
    for L in layers:
        if L not in meta:
            continue
        info = meta[L]
        path = tdir / f"L{L}.bin"
        if not path.is_file():
            raise AblitError(f"transplant manifest lists L{L} but {path} is missing")
        raw = path.read_bytes()
        if len(raw) != int(info["nbytes"]):
            raise AblitError(
                f"transplant L{L}: expected {info['nbytes']} bytes, got {len(raw)} "
                "— re-run ablit/fetch_transplant.py")
        if info["dtype"] != "BF16":
            raise AblitError(f"transplant L{L}: unsupported dtype {info['dtype']}")
        t = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(
            tuple(info["shape"]))
        out[L] = t
    if not out:
        raise AblitError(
            f"no transplant tensors under {tdir} cover ABLIT_LAYERS — "
            "run ablit/fetch_transplant.py first")
    return out


def apply_transplant(
    model: Any,
    donors: dict[int, torch.Tensor],
    layers: list[int],
    include_mtp: bool,
) -> dict[str, Any]:
    """Replace stock o_proj weights with the donor tensors (per-TP shard)."""
    text_model = unwrap_text_model(model)
    candidates = walk_o_proj(text_model)
    want = set(layers)
    world, rank = _tp_world(), _tp_rank()
    report: dict[str, Any] = {"edited_layers": [], "skipped": [], "mtp_edited": False,
                              "deltas": {}}
    seen_ids: set[int] = set()
    for name, idx, mod in candidates:
        if id(mod) in seen_ids:
            continue
        seen_ids.add(id(mod))
        is_mtp = _MTP_O_PROJ_RE.search(name) is not None
        if is_mtp and not include_mtp:
            report["skipped"].append({"name": name, "reason": "include_mtp=0"})
            continue
        if idx not in want:
            continue
        if idx not in donors:
            raise AblitError(
                f"ABLIT_METHOD=transplant has no donor tensor for layer {idx} "
                "— fetch it with ablit/fetch_transplant.py")
        donor = donors[idx]
        weight = getattr(mod, "weight", None)
        if weight is None or not torch.is_tensor(weight) or weight.dim() != 2:
            raise AblitError(f"ablit transplant: {name} has no 2-D .weight")
        full_in = donor.shape[1]
        local_in = weight.shape[1]
        if donor.shape[0] != weight.shape[0] or full_in != local_in * world:
            raise AblitError(
                f"ablit transplant: {name} shape {tuple(weight.shape)} does not "
                f"match donor {tuple(donor.shape)} at TP={world}")
        shard = donor if world == 1 else donor[:, rank * local_in:(rank + 1) * local_in]
        # Donors come from torch.frombuffer on CPU; o_proj is already on the
        # worker GPU. Compare and copy on weight's device (proj path already
        # does r.to(weight.device); transplant forgot).
        shard = shard.to(device=weight.device, dtype=weight.dtype)
        with torch.no_grad():
            w32 = weight.data.float()
            d32 = shard.float()
            rel_l2 = float((d32 - w32).norm() / w32.norm().clamp_min(1e-9))
            mean_rel = float((d32 - w32).abs().mean() / w32.abs().mean().clamp_min(1e-9))
            weight.data.copy_(shard)
        report["edited_layers"].append(idx)
        report["deltas"][idx] = {"rel_l2": rel_l2, "mean_rel": mean_rel}
        if is_mtp:
            report["mtp_edited"] = True
        logger.info(
            "ablit: transplanted %s (donor L%d) shape=%s rel_l2=%.4f mean_rel=%.4f",
            name, idx, tuple(weight.shape), rel_l2, mean_rel)
    report["edited_layers"].sort()
    return report


def maybe_apply(model: Any) -> dict[str, Any] | None:
    """Hook entrypoint. No-op unless ABLIT=1; raises AblitError if enabled
    but the recipe cannot be honored (fail loud beats silent stock weights)."""
    if not env_flag("ABLIT", False):
        return None

    ablit_dir = os.environ.get("ABLIT_DIR") or DEFAULT_ABLIT_DIR
    layers_spec = os.environ.get("ABLIT_LAYERS") or "15-45"
    include_mtp = env_flag("ABLIT_INCLUDE_MTP", True)
    method = (os.environ.get("ABLIT_METHOD") or "auto").strip().lower()
    layers = parse_layers(layers_spec)

    if method in ("auto", "transplant"):
        tdir = Path(ablit_dir) / TRANSPLANT_SUBDIR
        have_transplant = (tdir / "MANIFEST.json").is_file() and (
            method == "transplant"
            or any((tdir / f"L{L}.bin").is_file() for L in layers)
        )
        if have_transplant:
            if method == "auto":
                logger.info("ablit: ABLIT_METHOD=auto -> transplant "
                            "(ablit/transplant/ present; direction proj does "
                            "not align with stock o_proj on this model)")
            donors = load_transplant_tensors(ablit_dir, layers)
            report = apply_transplant(model, donors, layers, include_mtp)
            if not report["edited_layers"] and not report["mtp_edited"]:
                raise AblitError(
                    "ABLIT=1 transplant matched no o_proj — ABLIT_LAYERS="
                    f"{layers_spec} matched nothing under this model")
            deltas = list(report["deltas"].values())
            logger.info(
                "ablit: ON method=transplant layers=%s edited=%s mtp=%s "
                "mean rel_l2=%.4f (skipped=%d) — early safety-anchor layers "
                "stay stock",
                layers_spec,
                report["edited_layers"],
                report["mtp_edited"],
                sum(d["rel_l2"] for d in deltas) / max(len(deltas), 1),
                len(report["skipped"]),
            )
            return report
        if method == "transplant":
            raise AblitError(
                "ABLIT_METHOD=transplant but ablit/transplant/ is missing or "
                "empty — run: python3 ablit/fetch_transplant.py")
        logger.info("ablit: ABLIT_METHOD=auto -> proj (no ablit/transplant/ "
                    "tensors found — falling back to direction orthogonalization)")

    # ABLIT_METHOD=proj (or auto fallback): direction orthogonalization
    direction = os.environ.get("ABLIT_DIRECTION") or "dealign"
    try:
        alpha = float(os.environ.get("ABLIT_ALPHA") or "3.0")
    except ValueError as exc:
        raise AblitError(f"ABLIT_ALPHA is not a number: {exc}") from exc
    if not math.isfinite(alpha) or alpha <= 0:
        raise AblitError(f"ABLIT_ALPHA must be a positive finite number, got {alpha}")

    path = resolve_direction_path(ablit_dir, direction)
    if not path.is_file():
        raise AblitError(f"ABLIT=1 but direction file missing: {path}")
    r = load_direction(path)

    report = apply_ablit(model, r, layers, alpha, include_mtp)

    if not report["edited_layers"] and not report["mtp_edited"]:
        raise AblitError(
            "ABLIT=1 but no o_proj was edited — ABLIT_LAYERS="
            f"{layers_spec} matched nothing under this model"
        )
    logger.info(
        "ablit: ON method=proj direction=%s (%s) alpha=%s layers=%s edited=%s "
        "mtp=%s (skipped=%d) — early safety-anchor layers stay stock",
        direction,
        path.name,
        alpha,
        layers_spec,
        report["edited_layers"],
        report["mtp_edited"],
        len(report["skipped"]),
    )
    return report
