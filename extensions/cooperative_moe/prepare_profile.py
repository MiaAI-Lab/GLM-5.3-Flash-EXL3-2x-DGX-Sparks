"""Create an exclusive opt-in overlay from pinned source-built artifacts.

Does not download, install, change defaults, or restart anything. The native
ELF is accepted by source provenance plus ABI validation, not by an exact
compiler-output hash: CUDA and host toolchain updates may legitimately change
that hash. The DS4.1 cooperative library must not be substituted.
"""

import argparse
import hashlib
import json
import struct
from pathlib import Path, PurePosixPath

# Dense EXL3 leaves the cooperative routed-expert ABI and pointer tables
# unchanged. Regenerate from this source so a dense pack cannot select an
# older overlay that ignores its non_routed_exl3 declarations.
STOCK_SHA = "241ebf56a01ddf940ba7015f31355a1dad810b391f9d77ef79fa978f42ccbab3"
ADAPTER_SHA = "a81b5bd0d90ac6e41a9a32352df79f6be901d41296029a16e39a3eae1364aa4e"
UPSTREAM_REPOSITORY = "https://github.com/turboderp-org/exllamav3.git"
UPSTREAM_PIN = "02aef45cd681b960a00afcd0749a4ab99e6c1bfe"
SOURCE_DATE_EPOCH = 1789312052
COMBINED_TREE_SHA = "db5ecd53c27c5105bda1183c40c059cbf6286197b020b2acbdfd1a9b8490c4fc"
EXPECTED_INPUTS = {
    "native/cooperative_moe.cu": "b3716b1ebc0578611a4f2a014faa315c407fc599c6554a82e4ced9f05e0dec4a",
    "native/cooperative_moe_kernel.cuh": "b6c3fd1bb7abb996e977d92befa7e352d0c24772f1ea020c19670773df3a5cbb",
    "native/exl3_moe_coop.cuh": "62903607abe42a9d662ac101af91823dcd20f9935b6097a6708602683a5ccec8",
    "runtime.py": ADAPTER_SHA,
    "build.sh": "cd5e3de833dc1ee28b86d1dca94975fd89143e60ef73782400633616a1eb3f8a",
}
EXPECTED_FLAGS = [
    "-std=c++17", "-O3", "--use_fast_math", "-lineinfo",
    "--expt-relaxed-constexpr", "--frandom-seed", "0x474c4d3533434f4f",
    "-gencode", "arch=compute_121a,code=sm_121a", "-shared",
    "-Xcompiler", "-fPIC", "--ptxas-options=-v",
]
REQUIRED_NATIVE_SYMBOLS = (b"glm53_coop_abi", b"glm53_coop_info", b"glm53_coop_launch")


def checked(path, digest):
    data = path.read_bytes()
    got = hashlib.sha256(data).hexdigest()
    if digest.startswith("UNVALIDATED") or got != digest:
        raise ValueError(f"Unvalidated source/adapter hash: {path}")
    return data


def check_native_elf(path):
    """Reject wrong-architecture and obviously wrong cooperative libraries.

    Full runtime compatibility is checked by runtime.py through ABI, native
    layout, occupancy and both-GPU integration tests. This deliberately does
    not compare the complete ELF hash.
    """
    data = path.read_bytes()
    if len(data) < 64 or data[:6] != b"\x7fELF\x02\x01":
        raise ValueError(f"Not a 64-bit little-endian ELF: {path}")
    if struct.unpack_from("<H", data, 18)[0] != 183:  # EM_AARCH64
        raise ValueError(f"Not an AArch64 ELF: {path}")
    missing = [symbol.decode() for symbol in REQUIRED_NATIVE_SYMBOLS if symbol not in data]
    if missing:
        raise ValueError(f"Missing cooperative ABI symbols in {path}: {missing}")


def check_build_manifest(path):
    try:
        manifest = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid cooperative build manifest: {path}") from exc
    if manifest.get("schema") != 1 or manifest.get("artifact") != "glm53-cooperative-moe":
        raise ValueError(f"Unsupported cooperative build manifest: {path}")
    if manifest.get("target") != "sm_121a":
        raise ValueError(f"Wrong cooperative build target: {manifest.get('target')!r}")
    upstream = manifest.get("upstream") or {}
    if upstream.get("repository") != UPSTREAM_REPOSITORY:
        raise ValueError(f"Wrong ExLlamaV3 repository: {upstream.get('repository')!r}")
    if upstream.get("commit") != UPSTREAM_PIN:
        raise ValueError(f"Wrong ExLlamaV3 source pin: {upstream.get('commit')!r}")
    if upstream.get("tree_sha256") != COMBINED_TREE_SHA:
        raise ValueError("Cooperative build input tree does not match the reviewed sources")
    if manifest.get("inputs") != EXPECTED_INPUTS:
        raise ValueError("Cooperative native/adapter source hashes do not match")
    build = manifest.get("build") or {}
    if build.get("source_date_epoch") != SOURCE_DATE_EPOCH:
        raise ValueError("Cooperative build did not use the reviewed source epoch")
    if build.get("random_seed") != "0x474c4d3533434f4f":
        raise ValueError("Cooperative build did not use the reproducible CUDA random seed")
    if build.get("flags") != EXPECTED_FLAGS:
        raise ValueError("Cooperative build flags do not match the reviewed recipe")
    nvcc = build.get("nvcc")
    nvcc_sha = build.get("nvcc_sha256")
    if (
        not isinstance(nvcc, str)
        or "Cuda compilation tools" not in nvcc
        or not isinstance(nvcc_sha, str)
        or len(nvcc_sha) != 64
        or any(c not in "0123456789abcdef" for c in nvcc_sha)
        or not isinstance(build.get("host_compiler"), str)
        or not build["host_compiler"]
        or not isinstance(build.get("system"), dict)
        or set(build["system"]) != {"sysname", "release", "version", "machine"}
        or not all(isinstance(value, str) and value for value in build["system"].values())
    ):
        raise ValueError("Cooperative build manifest lacks compiler/system provenance")
    output = manifest.get("output") or {}
    binary = path.parent / "cooperative_moe.so"
    if output.get("sha256") != hashlib.sha256(binary.read_bytes()).hexdigest():
        raise ValueError("Cooperative binary does not match its build manifest")
    if output.get("size") != binary.stat().st_size:
        raise ValueError("Cooperative binary size does not match its build manifest")
    return manifest


def make_profile(stock, artifacts, runtime_directory, output):
    base = checked(Path(stock), STOCK_SHA)
    artifacts = Path(artifacts)
    checked(artifacts / "runtime.py", ADAPTER_SHA)
    check_build_manifest(artifacts / "build-manifest.json")
    check_native_elf(artifacts / "cooperative_moe.so")
    runtime_root = PurePosixPath(runtime_directory)
    if not runtime_root.is_absolute() or ".." in runtime_root.parts:
        raise ValueError(
            "Use an absolute container runtime directory without parent traversal"
        )
    footer = (
        "\n# Explicit fixed cooperative MoE opt-in; unsupported calls stay stock.\n"
        "import runpy as _coop_runpy\nimport sys as _coop_sys\n"
        f'_coop_setup = _coop_runpy.run_path({str(runtime_root / "runtime.py")!r})\n'
        f'_coop_setup["install"](_coop_sys.modules[__name__], library_root={str(runtime_root)!r}, enabled=True)\n'
    )
    with Path(output).open("xb") as handle:
        handle.write(base + footer.encode())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stock", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument(
        "--runtime-directory",
        required=True,
        help="Container path containing the verified binary and adapter on BOTH ranks",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    make_profile(args.stock, args.artifacts, args.runtime_directory, args.output)
    print(f"Wrote opt-in overlay: {args.output}; no service changes made")
