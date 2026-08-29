# Align GLM-5.3 video prompt placeholders to encoder grid_t.
# glm4_1v builds one image-block per timestamp; a 1s clip can yield 3 timestamps
# while video_grid_thw T=1, so vLLM assigns 100 encoder tokens to 300 slots.
#
# Glm5NextVideoProcessor has no max_duration. TRANSFORMERS_WITH_GA + a Glm4v
# processor wrapper still routed 5.3 through _get_video_second_idx_glm4v.
from __future__ import annotations

import sys
from pathlib import Path


KPOOL = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/"
    "sparse_attn_indexer_kpool.py"
)
KPOOL_OLD = "if current_platform.is_cuda() and select_k in (512, 1024, 2048):"
KPOOL_NEW = (
    "if False and current_platform.is_cuda() and "
    "select_k in (512, 1024, 2048):  # GB10 persistent_topk smem"
)


def _align_timestamps(timestamps: list, t_groups: int) -> list:
    if t_groups < 1:
        t_groups = 1
    if not timestamps:
        return [0] * t_groups
    n = len(timestamps)
    if n == t_groups:
        return timestamps
    if t_groups == 1:
        return [timestamps[0]]
    if n < t_groups:
        return timestamps + [timestamps[-1]] * (t_groups - n)
    last = n - 1
    return [timestamps[int(round(i * last / (t_groups - 1)))] for i in range(t_groups)]


def _use_glm4v_timestamps(hf_processor) -> bool:
    """True only for classic Glm4v video processors that expose max_duration."""
    from vllm.model_executor.models.glm4_1v import Glm4vProcessor

    name = type(hf_processor).__name__
    vp = getattr(hf_processor, "video_processor", None)
    vp_name = type(vp).__name__ if vp is not None else ""
    if "Glm5" in name or "Glm5" in vp_name:
        return False
    if vp is not None and not hasattr(vp, "max_duration"):
        return False
    return isinstance(hf_processor, Glm4vProcessor)


def apply() -> None:
    from vllm.model_executor.models.glm4_1v import Glm4vProcessingInfo

    if getattr(Glm4vProcessingInfo, "_glm53_video_t_aligned", False):
        return

    def _construct_video_placeholder(self, video_array, metadata, grid_thw):
        hf_processor = self.get_hf_processor()
        tokenizer = self.get_tokenizer()
        image_processor = hf_processor.image_processor
        hf_config = self.get_hf_config()
        merge_length = image_processor.merge_size**2
        t_hw = grid_thw.reshape(-1)
        t_groups, height, width = int(t_hw[0]), int(t_hw[1]), int(t_hw[2])
        n_per = int(height * width) // merge_length

        if _use_glm4v_timestamps(hf_processor):
            timestamps = self._get_video_second_idx_glm4v(metadata, len(video_array))
            ts_fmt = "{}"
        elif self._is_glmga_model(hf_processor):
            timestamps = self._get_video_second_idx_glmga(metadata, len(video_array))
            ts_fmt = "{:.1f} seconds"
        else:
            # Glm5NextProcessor / Glm5NextVideoProcessor: glm46v timestamps.
            timestamps = self._get_video_second_idx_glm46v(metadata, len(video_array))
            ts_fmt = "{:.1f} seconds"

        timestamps = _align_timestamps(list(timestamps), t_groups)
        embed_id = self._get_video_frame_embed_token_id(hf_processor)
        placeholder = [hf_config.video_start_token_id]
        for ts in timestamps:
            placeholder.append(hf_config.image_start_token_id)
            placeholder.extend([embed_id] * n_per)
            placeholder.append(hf_config.image_end_token_id)
            placeholder.extend(
                tokenizer.encode(ts_fmt.format(ts), add_special_tokens=False)
            )
        placeholder.append(hf_config.video_end_token_id)
        return placeholder

    Glm4vProcessingInfo._construct_video_placeholder = _construct_video_placeholder
    Glm4vProcessingInfo._glm53_video_t_aligned = True
    print(
        "glm53: video placeholders aligned to encoder grid_t (Glm5Next→glm46v)",
        file=sys.stderr,
    )


def _install_import_hook() -> None:
    import builtins
    import importlib

    if getattr(builtins, "_glm53_video_hook", False):
        return
    builtins._glm53_video_hook = True
    real_import = builtins.__import__
    applying = False

    def _maybe_apply(name: str) -> None:
        nonlocal applying
        if applying or "glm4_1v" not in name:
            return
        applying = True
        try:
            apply()
        except Exception as exc:
            print(f"glm53: video apply() failed: {exc!r}", file=sys.stderr)
        finally:
            applying = False

    def _import(name, globals=None, locals=None, fromlist=(), level=0):
        mod = real_import(name, globals, locals, fromlist, level)
        _maybe_apply(name)
        return mod

    builtins.__import__ = _import
    real_im = importlib.import_module

    def _im(name, package=None):
        mod = real_im(name, package)
        _maybe_apply(name)
        return mod

    importlib.import_module = _im


def _disable_gb10_persistent_topk() -> None:
    """Decode-path persistent_topk oversubscribes GB10 smem on long seqs."""
    if not KPOOL.is_file():
        raise FileNotFoundError(f"glm53: missing {KPOOL}")
    text = KPOOL.read_text()
    if KPOOL_NEW in text:
        print("glm53: persistent_topk already disabled in kpool", file=sys.stderr)
    elif KPOOL_OLD in text:
        KPOOL.write_text(text.replace(KPOOL_OLD, KPOOL_NEW, 1))
        print(
            "glm53: disabled GB10 persistent_topk (use top_k_per_row_decode)",
            file=sys.stderr,
        )
    else:
        raise RuntimeError(
            "glm53: kpool persistent_topk pattern not found — patch the file by hand"
        )
    cache = KPOOL.parent / "__pycache__"
    if cache.is_dir():
        for pyc in cache.glob("sparse_attn_indexer_kpool*.pyc"):
            pyc.unlink(missing_ok=True)


_install_import_hook()
try:
    apply()
except Exception:
    pass


if __name__ == "__main__":
    import shutil

    src = Path(__file__).resolve()
    dst = Path("/usr/local/lib/python3.12/dist-packages/glm53_video_patch.py")
    shutil.copy(src, dst)
    Path("/usr/local/lib/python3.12/dist-packages/glm53_video.pth").write_text(
        "import glm53_video_patch\n"
    )
    _disable_gb10_persistent_topk()
    print("glm53: overlay install ok aligned=True", file=sys.stderr)
