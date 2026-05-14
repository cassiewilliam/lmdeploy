#!/usr/bin/env python3
# Copyright (c) OpenMMLab. All rights reserved.

"""Build LMDeploy's patched FlashInfer SM100 groupwise GEMM JIT module.

FlashInfer 0.6.11's small-batch FP8 groupwise path supplies CUTLASS
KernelHardwareInfo before can_implement(), while the regular path does not.
On B200 this can reject otherwise valid large-M dense GEMMs.  Keep using the
upstream FlashInfer sources, but shadow the one header with the same
KernelHardwareInfo plumbing for the regular path.
"""

import os
import shutil
import sys
from itertools import product
from pathlib import Path

import jinja2
import torch

from flashinfer.jit import env as jit_env
from flashinfer.jit.core import current_compilation_context, gen_jit_spec
from flashinfer.jit.utils import dtype_cutlass_map, filename_safe_dtype_map, write_if_different


def _patch_groupwise_header(gen_directory: Path) -> Path:
    include_root = gen_directory / "patched_include"
    source_root = jit_env.FLASHINFER_INCLUDE_DIR / "flashinfer"
    target_root = include_root / "flashinfer"
    shutil.copytree(source_root, target_root, dirs_exist_ok=True)

    header = target_root / "gemm" / "gemm_groupwise_sm100.cuh"
    text = header.read_text()
    old = """                                     {
                                         {},  // epilogue.thread
                                         D_ptr,
                                         stride_C,
                                         D_ptr,
                                         stride_C,
                                     }};"""
    new = """                                     {
                                         {},  // epilogue.thread
                                         D_ptr,
                                         stride_C,
                                         D_ptr,
                                         stride_C,
                                     },
                                     []() {
                                       auto hw_info =
                                           cutlass::KernelHardwareInfo::make_kernel_hardware_info<GemmKernel>();
                                       hw_info.cluster_shape = {MmaSM, 1, 1};
                                       hw_info.cluster_shape_fallback = {MmaSM, 1, 1};
                                       return hw_info;
                                     }()};"""
    if old not in text:
        if "make_kernel_hardware_info<GemmKernel>()" not in text:
            raise RuntimeError("FlashInfer groupwise SM100 header layout changed; patch did not apply")
    else:
        text = text.replace(old, new, 1)
        header.write_text(text)

    return include_root


def gen_lmdeploy_gemm_sm100_module():
    gen_directory = jit_env.FLASHINFER_GEN_SRC_DIR / "gen_lmdeploy_gemm_sm100"
    os.makedirs(gen_directory, exist_ok=True)

    patched_include = _patch_groupwise_header(gen_directory)

    source_paths = []
    for prefix in ["gemm_groupwise", "group_gemm_fp8_groupwise"]:
        with open(jit_env.FLASHINFER_CSRC_DIR / f"{prefix}_sm100_kernel_inst.jinja") as f:
            kernel_inst_templ = jinja2.Template(f.read())
        dtype_in_list = [torch.float8_e4m3fn, torch.float8_e5m2]
        dtype_out_list = [torch.float16, torch.bfloat16]
        scale_major_k_list = ["true", "false"]
        mma_sm_list = [1, 2]
        for dtype_in, dtype_out, scale_major_k, mma_sm in product(
                dtype_in_list, dtype_out_list, scale_major_k_list, mma_sm_list):
            name_dtype_in = filename_safe_dtype_map[dtype_in]
            name_dtype_out = filename_safe_dtype_map[dtype_out]
            dest_path = (
                gen_directory
                / f"{prefix}_{name_dtype_in}_{name_dtype_out}_major{scale_major_k}_mma{mma_sm}_sm100.cu")
            source_paths.append(dest_path)
            source = kernel_inst_templ.render(
                dtype_in=dtype_cutlass_map[dtype_in],
                dtype_out=dtype_cutlass_map[dtype_out],
                scale_major_k=scale_major_k,
                mma_sm=mma_sm,
            )
            write_if_different(dest_path, source)

    prefix = "group_gemm_mxfp4_groupwise"
    with open(jit_env.FLASHINFER_CSRC_DIR / f"{prefix}_sm100_kernel_inst.jinja") as f:
        kernel_inst_templ = jinja2.Template(f.read())
    dtype_a_list = [torch.float8_e4m3fn, torch.float8_e5m2]
    dtype_d_list = [torch.float16, torch.bfloat16]
    mma_sm_list = [1, 2]
    swap_ab_list = ["true", "false"]
    for dtype_a, dtype_d, mma_sm, swap_ab in product(dtype_a_list, dtype_d_list, mma_sm_list, swap_ab_list):
        name_dtype_a = filename_safe_dtype_map[dtype_a]
        name_dtype_d = filename_safe_dtype_map[dtype_d]
        dest_path = (
            gen_directory / f"{prefix}_{name_dtype_a}_{name_dtype_d}_mma{mma_sm}_swap{swap_ab}_sm100.cu")
        source_paths.append(dest_path)
        source = kernel_inst_templ.render(
            dtype_a=dtype_cutlass_map[dtype_a],
            dtype_b="cutlass::float_e2m1_t",
            dtype_d=dtype_cutlass_map[dtype_d],
            mma_sm=mma_sm,
            swap_ab=swap_ab,
        )
        write_if_different(dest_path, source)

    for filename in [
            "gemm_groupwise_sm100.cu",
            "group_gemm_fp8_groupwise_sm100.cu",
            "group_gemm_mxfp4_groupwise_sm100.cu",
            "gemm_sm100_binding.cu",
            "group_gemm_sm100_binding.cu",
    ]:
        src_path = jit_env.FLASHINFER_CSRC_DIR / filename
        dest_path = gen_directory / filename
        source_paths.append(dest_path)
        with open(src_path) as f:
            write_if_different(dest_path, f.read())

    nvcc_flags = current_compilation_context.get_nvcc_flags_list(supported_major_versions=[10, 11, 12])
    return gen_jit_spec(
        "lmdeploy_gemm_sm100",
        source_paths,
        extra_cuda_cflags=nvcc_flags + ["-DCUTLASS_ENABLE_GDC_FOR_SM100=1"],
        extra_include_paths=[patched_include],
    )


def main() -> int:
    try:
        spec = gen_lmdeploy_gemm_sm100_module()
        if not spec.is_compiled:
            spec.build(os.environ.get("FLASHINFER_JIT_VERBOSE", "0") == "1")
        so_path = spec.get_library_path()
        if not so_path.exists():
            raise RuntimeError(f"reported path missing: {so_path}")
        sys.stdout.write(str(so_path.resolve()))
        return 0
    except Exception as exc:
        sys.stderr.write(f"[flashinfer-gemm] patched SM100 build failed: {type(exc).__name__}: {exc}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
