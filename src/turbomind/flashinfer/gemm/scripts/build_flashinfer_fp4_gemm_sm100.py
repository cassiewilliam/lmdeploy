#!/usr/bin/env python3
# Copyright (c) OpenMMLab. All rights reserved.

"""Build FlashInfer's SM10x CUTLASS FP4 GEMM JIT module."""

import argparse
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', choices=['sm100', 'sm103'], default='sm100')
    args = parser.parse_args()

    try:
        from flashinfer.gemm.gemm_base import gen_gemm_sm100_module_cutlass_fp4, gen_gemm_sm103_module_cutlass_fp4

        if args.arch == 'sm103':
            spec = gen_gemm_sm103_module_cutlass_fp4()
        else:
            spec = gen_gemm_sm100_module_cutlass_fp4()
        if not spec.is_compiled:
            spec.build(os.environ.get("FLASHINFER_JIT_VERBOSE", "0") == "1")
        so_path = spec.get_library_path()
        if not so_path.exists():
            raise RuntimeError(f"reported path missing: {so_path}")
        sys.stdout.write(str(so_path.resolve()))
        return 0
    except Exception as exc:
        sys.stderr.write(f"[flashinfer-gemm] {args.arch} FP4 build failed: {type(exc).__name__}: {exc}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
