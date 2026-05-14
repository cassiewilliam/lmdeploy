#!/usr/bin/env python3
# Copyright (c) OpenMMLab. All rights reserved.
"""Resolve the trtllm-gen FMHA `.so` path published by `flashinfer-python`.

Invoked by CMake at configure time (see `../CMakeLists.txt`) — the resolved
path is baked into a generated header so the C++ wrapper can `LoadFromFile`
it via tvm-ffi at runtime, without compiling any FlashInfer source itself.

Also usable standalone:
    python3 resolve_so.py            # prints path on stdout, exit 0
    python3 resolve_so.py --probe    # exit 0 if resolvable, else 1; no print

Resolution strategy:
  1. Import `flashinfer`.
  2. Walk a small list of known JIT entry points (FlashInfer's internal API
     for the trtllm-gen FMHA module has shifted across 0.6.x revisions) and
     use whichever resolves to a callable returning a module-like object
     with a `.lib_path` / `.so_path` / `.library_path` attribute.
  3. Fall back to scanning `flashinfer/jit/`'s on-disk cache for a `.so`
     whose name matches `*trtllm*fmha*.so` (lets us recover the path
     without re-running JIT once a previous build cached it).

Print empty string + exit 1 on failure.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Optional


def _import_flashinfer():
    try:
        import flashinfer  # noqa: F401
        return True
    except ImportError as e:
        sys.stderr.write(
            f"[resolve_so] flashinfer-python is not installed: {e}\n"
            "  Install FlashInfer in the build/runtime env; the cu130 Docker "
            "image installs published wheels via docker/install_flashinfer_wheel.sh.\n")
        return False


def _try_jit_entry_points() -> Optional[str]:
    """Try the known JIT entry-point shapes in order; return path or None."""
    candidates = [
        ("flashinfer.jit", "gen_trtllm_gen_fmha_module"),
        ("flashinfer.jit.trtllm_gen_fmha", "gen_trtllm_gen_fmha_module"),
        ("flashinfer.jit.trtllm_gen_fmha", "trtllm_gen_fmha_module"),
        ("flashinfer.fmha", "gen_trtllm_gen_fmha_module"),
        ("flashinfer.fmha", "trtllm_gen_fmha_module"),
    ]
    last_err: Optional[Exception] = None
    for mod_path, attr in candidates:
        try:
            module = __import__(mod_path, fromlist=[attr])
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
        fn = getattr(module, attr, None)
        if fn is None:
            continue
        try:
            obj = fn()
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
        for attr_name in ("so_path", "lib_path", "library_path"):
            p = getattr(obj, attr_name, None)
            if p:
                return str(p)
        if isinstance(obj, (str, os.PathLike)):
            return str(obj)
    if last_err is not None:
        sys.stderr.write(
            f"[resolve_so] JIT entry-point probe failed: {last_err}\n")
    return None


def _scan_jit_cache() -> Optional[str]:
    """Last-ditch: glob `~/.cache/flashinfer/**` for a JIT-compiled FMHA `.so`.

    Observed layouts (FlashInfer 0.6.11):
      ~/.cache/flashinfer/0.6.11/<arch>/cached_ops/fmha_gen/fmha_gen.so
      ~/.cache/flashinfer/0.6.11/<arch>/cached_ops/trtllm*fmha*/<name>.so
    Older revs: ~/.cache/flashinfer/<hash>/{trtllm_fmha,fmha}*.so
    """
    home = os.path.expanduser("~")
    roots = [
        os.path.join(home, ".cache", "flashinfer"),
        os.environ.get("FLASHINFER_CACHE_DIR", ""),
    ]
    patterns = (
        "**/cached_ops/fmha_gen/*.so",
        "**/cached_ops/*fmha*/*.so",
        "**/cached_ops/*trtllm*fmha*/*.so",
        "**/fmha_gen*.so",
        "**/trtllm*fmha*.so",
        "**/fmha*trtllm*.so",
    )
    for root in filter(None, roots):
        if not os.path.isdir(root):
            continue
        for pattern in patterns:
            hits = sorted(glob.glob(os.path.join(root, pattern), recursive=True),
                          key=os.path.getmtime, reverse=True)
            if hits:
                return hits[0]
    return None


def resolve() -> Optional[str]:
    if not _import_flashinfer():
        return None
    p = _try_jit_entry_points()
    if p and os.path.exists(p):
        return p
    p2 = _scan_jit_cache()
    if p2 and os.path.exists(p2):
        return p2
    return None


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true",
                    help="exit 0 if resolvable; print nothing")
    args = ap.parse_args(argv[1:])
    p = resolve()
    if not p:
        return 1
    if not args.probe:
        sys.stdout.write(p)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
