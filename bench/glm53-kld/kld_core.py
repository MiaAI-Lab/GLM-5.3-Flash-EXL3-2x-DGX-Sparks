# SPDX-License-Identifier: Apache-2.0
"""Shared primitives for GLM-5.3-Flash quant-fidelity (KLD) runs.

Two protocols live here.

``load_distribution_suite`` reads the published suite that is actually on the
hub (``malaiwah/GLM-5.3-Flash-fidelity-suite-v1``, schema
``glm53flash-distribution-fidelity/6``): teacher-forced contexts, and hidden
states captured **after the final RMSNorm, before ``lm_head``** (semantic point
``after_final_rmsnorm_before_lm_head``), so replay multiplies by the shared head
and by nothing else. Read ``<lane>/capture-cut-point.json`` before writing any
scorer: applying ``head/final_norm.safetensors`` on top of these tensors is the
single most expensive mistake available here - it rescales every logit row and
inflates KLD while looking completely healthy.

``load_token_panel`` keeps the older field-compatible token-panel/logits shape
(brandonmusic/GLM-5.3-Flash-tr3-4bpw discussion #1) for full-vocabulary logits
captures.

Nothing here is copied from either (source-available, non-standard-licensed)
repository; the schemas are field-compatible so a receipt produced here can be
fed to - or by - their measurement scripts, but the code is ours.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

CAPTURE_SCHEMA = "quant-pipeline.glm53-logit-capture.v1"
PANEL_SCHEMA = "quant-pipeline.glm53-token-panel.v1"
PANEL_RECEIPT_SCHEMA = "quant-pipeline.glm53-token-panel-receipt.v1"
LOGITS_FIELD = "logits"


# ---------------------------------------------------------------- sealing ---


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path: str | Path, data: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def write_json(path: str | Path, value: Any) -> str:
    data = canonical_json(value)
    atomic_write(path, data)
    return sha256_bytes(data)


def seal(payload: dict, seal_field: str) -> dict:
    """Attach ``payload[seal_field]`` = sha256 of the payload without it."""
    body = {key: value for key, value in payload.items() if key != seal_field}
    payload[seal_field] = sha256_bytes(canonical_json(body))
    return payload


def unsealed(payload: dict, seal_field: str, label: str) -> dict:
    body = {key: value for key, value in payload.items() if key != seal_field}
    if payload.get(seal_field) != sha256_bytes(canonical_json(body)):
        raise ValueError(f"{label} seal mismatch")
    return body


def prepare_empty_destination(path: str | Path) -> Path:
    destination = Path(path)
    if destination.exists():
        if not destination.is_dir():
            raise ValueError(f"destination exists and is not a directory: {destination}")
        if next(destination.iterdir(), None) is not None:
            raise FileExistsError(f"destination is not empty; choose a new path: {destination}")
    else:
        destination.mkdir(parents=True)
    return destination


# ------------------------------------------------------------------- math ---


def token_kld(teacher_logits: np.ndarray, student_logits: np.ndarray) -> np.ndarray:
    """Per-token KL(teacher || student) in nats, float64, one row per position."""
    teacher = np.asarray(teacher_logits, dtype=np.float64)
    student = np.asarray(student_logits, dtype=np.float64)
    if teacher.shape != student.shape or teacher.ndim != 2 or teacher.shape[1] <= 1:
        raise ValueError("logit matrices must be 2-D (positions, vocab) and match")
    if not (np.isfinite(teacher).all() and np.isfinite(student).all()):
        raise ValueError("logits must be finite")
    teacher_logp = _log_softmax(teacher)
    student_logp = _log_softmax(student)
    values = np.sum(np.exp(teacher_logp) * (teacher_logp - student_logp), axis=-1)
    if not np.isfinite(values).all():
        raise ValueError("KLD result is non-finite")
    return values


def _log_softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=-1, keepdims=True)
    return shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))


def summarize(values: np.ndarray) -> dict[str, float | int]:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.size == 0 or not np.isfinite(vector).all():
        raise ValueError("cannot summarize an empty or non-finite vector")
    ordered = np.sort(vector)
    tail = ordered[max(0, int(np.floor(0.95 * vector.size))) :]
    return {
        "count": int(vector.size),
        "mean": float(np.mean(vector)),
        "std": float(np.std(vector, ddof=1)) if vector.size > 1 else 0.0,
        "p50": float(np.quantile(vector, 0.50)),
        "p95": float(np.quantile(vector, 0.95)),
        "p99": float(np.quantile(vector, 0.99)),
        "cvar95": float(np.mean(tail)),
        "max": float(ordered[-1]),
    }


# -------------------------------------------------------------- artifacts ---


@dataclass(frozen=True)
class PanelWindow:
    window_id: str
    document_id: str
    domain: str
    role: str
    token_ids: np.ndarray
    token_ids_sha256: str
    attention_mask_sha256: str
    prediction_positions: int


def _artifact_map(rows: Any) -> dict[str, Path]:
    if not isinstance(rows, list) or not rows:
        raise ValueError("receipt must carry a nonempty 'artifacts' list")
    resolved: dict[str, Path] = {}
    for row in rows:
        if not isinstance(row, dict) or not {"path", "bytes", "sha256"} <= set(row):
            raise ValueError("artifact rows need path, bytes and sha256")
        path = Path(str(row["path"]))
        if not path.is_file():
            raise FileNotFoundError(f"sealed artifact missing: {path}")
        if path.stat().st_size != int(row["bytes"]) or sha256_file(path) != row["sha256"]:
            raise ValueError(f"sealed artifact content mismatch: {path}")
        resolved[str(row["sha256"])] = path
    return resolved


def load_token_panel(
    receipt_path: str | Path,
    *,
    roles: Iterable[str] = ("final",),
    vocab_size: int | None = None,
) -> tuple[dict, list[PanelWindow]]:
    """Read a published token-panel receipt (.npy token/mask artifacts)."""
    receipt_path = Path(receipt_path).resolve()
    receipt = json.loads(receipt_path.read_text())
    unsealed(receipt, "receipt_sha256", "token-panel receipt")
    if receipt.get("schema") != PANEL_RECEIPT_SCHEMA:
        raise ValueError(f"unexpected panel receipt schema: {receipt.get('schema')}")
    artifacts = _artifact_map(receipt.get("artifacts"))
    panel_digest = receipt.get("token_panel_artifact_sha256")
    if panel_digest not in artifacts:
        raise ValueError("token-panel receipt does not resolve its panel artifact")
    panel = json.loads(artifacts[panel_digest].read_text())
    if panel.get("schema") != PANEL_SCHEMA:
        raise ValueError("token-panel artifact has the wrong schema")
    requested = {str(role) for role in roles}
    windows: list[PanelWindow] = []
    for row in panel.get("windows", ()):
        if row.get("role") not in requested:
            continue
        token_digest = row["token_ids_sha256"]
        mask_digest = row["attention_mask_sha256"]
        tokens = _read_npy_int(artifacts[token_digest])
        mask = _read_npy_int(artifacts[mask_digest]).astype(np.int64)
        if tokens.shape != mask.shape:
            raise ValueError("panel token IDs and attention mask are misaligned")
        if vocab_size is not None and (int(tokens.min()) < 0 or int(tokens.max()) >= vocab_size):
            raise ValueError("panel token ID is outside the released vocabulary")
        positions = int(np.count_nonzero(mask[:-1] & mask[1:]))
        if positions <= 0 or row.get("prediction_positions") != positions:
            raise ValueError("panel prediction-position count differs from the causal mask")
        windows.append(
            PanelWindow(
                window_id=str(row["window_id"]),
                document_id=str(row["document_id"]),
                domain=str(row["domain"]),
                role=str(row["role"]),
                token_ids=tokens,
                token_ids_sha256=token_digest,
                attention_mask_sha256=mask_digest,
                prediction_positions=positions,
            )
        )
    if not windows:
        raise ValueError(f"token panel has no windows for roles {sorted(requested)}")
    if len({window.window_id for window in windows}) != len(windows):
        raise ValueError("token-panel window identities are not unique")
    return receipt, windows


def _read_npy_int(path: Path) -> np.ndarray:
    if path.suffix != ".npy":
        raise ValueError(f"panel artifacts must be .npy, got {path}")
    values = np.load(path)
    if values.dtype.kind not in "iub":
        raise ValueError(f"unexpected panel dtype {values.dtype} in {path}")
    return values.astype(np.int64, copy=False)


def load_capture_receipt(path: str | Path, *, expected_role: str | None = None) -> dict:
    """Read a capture receipt and verify every sealed logit file on disk."""
    receipt = json.loads(Path(path).read_text())
    unsealed(receipt, "receipt_sha256", "logit capture receipt")
    if receipt.get("schema") != CAPTURE_SCHEMA:
        raise ValueError(f"unexpected capture schema: {receipt.get('schema')}")
    if expected_role is not None and receipt.get("capture_role") != expected_role:
        raise ValueError(f"capture role is {receipt.get('capture_role')!r}, expected {expected_role!r}")
    for row in receipt.get("logit_files", ()):
        logit_path = Path(row["path"])
        if not logit_path.is_file():
            raise FileNotFoundError(f"captured logits missing: {logit_path}")
        if logit_path.stat().st_size != int(row["bytes"]) or sha256_file(logit_path) != row["sha256"]:
            raise ValueError(f"captured logits tampered or stale: {logit_path}")
    return receipt


# -------------------------------------------- distribution-fidelity suite ----

SUITE_MANIFEST_SCHEMA = "glm53flash-distribution-fidelity/6"
CAPTURE_MANIFEST_SCHEMA = "glm53flash-fidelity-capture/2"
CUT_POINT_SCHEMA = "malaiwah.glm53-capture-cut-point.v1"
#: The published hidden states are already past the final RMSNorm. Replay
#: multiplies by lm_head and by nothing else; see the module docstring.
SEMANTIC_POINT = "after_final_rmsnorm_before_lm_head"
SUITE_TENSOR_KEY = "hidden_states"
HEAD_TENSOR_KEY = "lm_head.weight"

#: KLD between two runs of the *same* model, measured by the suite on its 32
#: sentinel contexts. Anything below this is engine jitter, not quantization.
PUBLISHED_ENGINE_NOISE_FLOOR_NATS = {
    "bf16": 0.0008700157463937287,
    "fp8": 0.0006256537334864121,
}


def token_ids_sha256(token_ids: Iterable[int]) -> str:
    """The digest the suite uses for one window: json.dumps defaults, UTF-8."""
    return sha256_bytes(json.dumps([int(token) for token in token_ids]).encode())


def single_tensor(path, prefer: tuple[str, ...]) -> "torch.Tensor":
    """Load the one tensor in a safetensors file, whatever it is called.

    The published files name their single tensor ``hidden_states`` (lanes) and
    ``weight`` (head, final norm); our own captures write ``hidden``. Selecting by
    position - exactly one tensor, asserted here - beats trusting a name, and the
    shape/dtype checks downstream are what actually pin the semantics.
    """
    import torch
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as handle:
        keys = list(handle.keys())
        if len(keys) != 1:
            raise ValueError(f"{path}: expected exactly one tensor, got {sorted(keys)}")
        chosen = next((key for key in prefer if key in keys), keys[0])
        return handle.get_tensor(chosen)


def load_lane_tensor(path, prefer: tuple[str, ...] = (SUITE_TENSOR_KEY, "hidden")) -> "torch.Tensor":
    return single_tensor(path, prefer)


def load_head(path) -> "torch.Tensor":
    return single_tensor(path, (HEAD_TENSOR_KEY, "weight"))


@dataclass(frozen=True)
class SuiteContext:
    index: int
    stratum: str
    partition: str
    source_cluster: str
    rows: int
    tokens: np.ndarray
    targets: np.ndarray
    hidden_path: Path
    token_sha256: str
    hidden_sha256: str | None


def _read_token_json(path: Path) -> list[int]:
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        token_ids = payload
    elif isinstance(payload, dict):
        for key in ("token_ids", "tokens", "input_ids", "ids"):
            if isinstance(payload.get(key), list):
                token_ids = payload[key]
                break
        else:
            raise ValueError(f"{path}: no token id array under token_ids/tokens/input_ids/ids")
    else:
        raise ValueError(f"{path}: unexpected token artifact type {type(payload).__name__}")
    if not token_ids:
        raise ValueError(f"{path}: empty token window")
    return [int(token) for token in token_ids]


def load_distribution_suite(
    suite_dir: str | Path,
    *,
    lane: str | Path = "reference-bf16-shard0",
    indices: Iterable[int] | None = None,
    partitions: Iterable[str] | None = None,
    strata: Iterable[str] | None = None,
    limit: int | None = None,
    # A capture run needs only the sealed token windows; making the teacher lane
    # optional there keeps our capture from depending on a download it never reads.
    require_lane_files: bool = True,
) -> tuple[dict, list[SuiteContext]]:
    """Read sealed windows plus one capture lane, verifying what is on disk.

    ``lane`` is a directory name inside ``suite_dir`` (the BF16 reference or the
    FP8 as-served lane) or a path to a lane we captured ourselves. Both carry
    ``capture-manifest-shard.json``, which - not ``capture-manifest-full.json`` -
    describes the bytes actually present: the full manifest advertises the whole
    5,120-context run inside a directory holding 512 files, so coverage computed
    from it is wrong by ten times.
    """
    suite_dir = Path(suite_dir).resolve()
    manifest_path = suite_dir / "suite" / "suite-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    schema = str(manifest.get("schema", ""))
    if not schema.startswith("glm53flash-distribution-fidelity/"):
        raise ValueError(f"unrecognised suite schema: {schema!r}")

    context_length = int(manifest["context_length"])
    rows = int(manifest["scored_positions_per_context"])
    if rows != context_length - 1:
        raise ValueError(
            f"scored_positions_per_context={rows} is not context_length-1={context_length - 1}; "
            "the row/target alignment assumed here no longer holds"
        )

    lane_dir = Path(lane)
    if not lane_dir.is_absolute():
        lane_dir = suite_dir / lane
    if not lane_dir.is_dir():
        raise FileNotFoundError(f"capture lane not present: {lane_dir}")

    shard_manifest_path = lane_dir / "capture-manifest-shard.json"
    published_sha: dict[int, str] = {}
    lane_meta: dict[str, Any] = {"shard_manifest": None}
    if shard_manifest_path.is_file():
        shard = json.loads(shard_manifest_path.read_text())
        if shard.get("schema") != CAPTURE_MANIFEST_SCHEMA:
            raise ValueError(f"unexpected capture schema: {shard.get('schema')!r}")
        if shard.get("semantic_point") != SEMANTIC_POINT:
            raise ValueError(
                f"lane semantic point is {shard.get('semantic_point')!r}, expected {SEMANTIC_POINT!r}"
            )
        declared_key = shard.get("tensor_key")
        if declared_key not in (SUITE_TENSOR_KEY, "hidden"):
            raise ValueError(f"lane tensor key {declared_key!r} is not a known hidden-state key")
        published_sha = {int(row["index"]): row["sha256"] for row in shard.get("captures", [])}
        lane_meta = {
            "shard_manifest": str(shard_manifest_path),
            "semantic_point": shard["semantic_point"],
            "tensor_key": declared_key,
            "lane": shard.get("lane"),
            "filter": shard.get("filter"),
            "declared_contexts": len(published_sha),
        }

    cut_point_path = lane_dir / "capture-cut-point.json"
    if cut_point_path.is_file():
        cut_point = json.loads(cut_point_path.read_text())
        if cut_point.get("semantic_point") != SEMANTIC_POINT:
            raise ValueError(f"cut point says {cut_point.get('semantic_point')!r}")
        if cut_point.get("final_norm", {}).get("applied_at_replay"):
            raise ValueError("cut point claims final norm is applied at replay; re-read it")
        lane_meta["cut_point_dtype"] = cut_point.get("dtype")

    wanted_indices = None if indices is None else {int(i) for i in indices}
    wanted_partitions = None if partitions is None else {str(p) for p in partitions}
    wanted_strata = None if strata is None else {str(s) for s in strata}

    contexts: list[SuiteContext] = []
    skipped: list[dict] = []
    for row in manifest.get("context_index", []):
        index = int(row["index"])
        if wanted_indices is not None and index not in wanted_indices:
            continue
        if wanted_partitions is not None and row.get("partition") not in wanted_partitions:
            continue
        if wanted_strata is not None and row.get("stratum") not in wanted_strata:
            continue
        hidden_path = lane_dir / f"hidden_{index:04d}.safetensors"
        if not hidden_path.is_file() and require_lane_files:
            skipped.append({"index": index, "reason": "hidden missing"})
            continue
        token_path = suite_dir / "suite" / row["file"]
        if not token_path.is_file():
            skipped.append({"index": index, "reason": f"token file missing: {token_path}"})
            continue
        tokens = _read_token_json(token_path)
        digest = token_ids_sha256(tokens)
        if row.get("token_sha256") and digest != row["token_sha256"]:
            raise ValueError(
                f"{token_path}: token digest {digest[:12]} != sealed {row['token_sha256'][:12]} "
                "(wrong tokenizer snapshot, or a corrupted download)"
            )
        if len(tokens) != context_length:
            raise ValueError(f"{token_path}: {len(tokens)} tokens, sealed context_length {context_length}")
        contexts.append(SuiteContext(
            index=index,
            stratum=str(row["stratum"]),
            partition=str(row.get("partition", "")),
            source_cluster=str(row.get("source_cluster", "")),
            rows=rows,
            tokens=np.asarray(tokens, dtype=np.int64),
            # Row r is the state after consuming tokens[r]; it predicts tokens[r+1].
            targets=np.asarray(tokens[1:rows + 1], dtype=np.int64),
            hidden_path=hidden_path,
            token_sha256=digest,
            hidden_sha256=published_sha.get(index),
        ))
        if limit is not None and len(contexts) >= limit:
            break
    if not contexts:
        raise ValueError(f"no complete contexts in lane {lane_dir} ({len(skipped)} skipped)")

    meta = {
        "schema": schema,
        "suite_dir": str(suite_dir),
        "lane_dir": str(lane_dir),
        "model": manifest.get("model"),
        "model_identity": manifest.get("model_identity"),
        "context_length": context_length,
        "scored_positions_per_context": rows,
        "hidden_size": int(manifest["hidden_size"]),
        "vocab_size": int(manifest["vocab_size"]),
        "suite_token_sha256": manifest.get("suite_token_sha256"),
        "sealed_contexts_total": len(manifest.get("context_index", [])),
        "lane": lane_meta,
        "skipped": skipped,
    }
    return meta, contexts


def load_replay_head(suite_dir: str | Path) -> tuple[Path, dict]:
    """The shared lm_head, with its published digest for the receipt."""
    suite_dir = Path(suite_dir)
    path = suite_dir / "head" / "head.safetensors"
    if not path.is_file():
        raise FileNotFoundError(path)
    meta: dict[str, Any] = {"path": str(path), "sha256": sha256_file(path)}
    extraction = suite_dir / "head" / "head-extraction.json"
    if extraction.is_file():
        published = json.loads(extraction.read_text())["tensors"]["head"]
        meta["source_tensor"] = published.get("source_tensor")
        meta["source_shard"] = published.get("source_shard")
        meta["published_sha256"] = published.get("sha256")
        if published.get("sha256") and published["sha256"] != meta["sha256"]:
            raise ValueError(
                f"{path}: digest {meta['sha256'][:12]} != published {published['sha256'][:12]}"
            )
    return path, meta


def cluster_bootstrap_ci(
    rows: list[dict],
    *,
    field: str = "mean_kld",
    cluster: str = "source_cluster",
    samples: int = 10000,
    seed: int = 20260901,
    confidence: float = 0.95,
) -> dict:
    """Context-mean CI by resampling *source clusters*, as the suite does.

    Windows drawn from one document are not independent observations, so
    resampling contexts (or positions) understates the interval. The published
    report bootstraps 837 source clusters with 10,000 draws.
    """
    groups: dict[str, list[float]] = {}
    for row in rows:
        groups.setdefault(str(row[cluster]), []).append(float(row[field]))
    if not groups:
        raise ValueError("no rows to bootstrap")
    keys = sorted(groups)
    # Resampling clusters makes the statistic sum(values)/sum(counts) over the
    # drawn clusters, so it reduces to two matrix-vector products per draw.
    sums = np.array([sum(groups[key]) for key in keys], dtype=np.float64)
    counts = np.array([len(groups[key]) for key in keys], dtype=np.float64)
    generator = np.random.default_rng(seed)
    picked = generator.integers(0, len(keys), size=(samples, len(keys)))
    weights = np.stack([np.bincount(draw, minlength=len(keys)) for draw in picked]).astype(np.float64)
    means = (weights @ sums) / (weights @ counts)
    overall = float(sums.sum() / counts.sum())
    alpha = 1.0 - confidence
    return {
        "mean": overall,
        "ci95_low": float(np.quantile(means, alpha / 2)),
        "ci95_high": float(np.quantile(means, 1 - alpha / 2)),
        "clusters": len(keys),
        "samples": samples,
        "seed": seed,
    }


def token_jsd_bits(teacher_logits: np.ndarray, student_logits: np.ndarray) -> np.ndarray:
    """Per-token Jensen-Shannon divergence in bits (base 2), float64."""
    teacher = np.asarray(teacher_logits, dtype=np.float64)
    student = np.asarray(student_logits, dtype=np.float64)
    if teacher.shape != student.shape or teacher.ndim != 2:
        raise ValueError("JSD matrices must be 2-D (positions, vocab) and match")
    p = np.exp(_log_softmax(teacher))
    q = np.exp(_log_softmax(student))
    mixture = 0.5 * (p + q)
    log_m = np.log(mixture, where=mixture > 0, out=np.zeros_like(mixture))
    ent_p = np.sum(np.where(p > 0, p * (np.log(p, where=p > 0, out=np.zeros_like(p)) - log_m), 0.0), axis=-1)
    ent_q = np.sum(np.where(q > 0, q * (np.log(q, where=q > 0, out=np.zeros_like(q)) - log_m), 0.0), axis=-1)
    return (ent_p + ent_q) * (0.5 / np.log(2.0))
