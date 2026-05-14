# Copyright (c) OpenMMLab. All rights reserved.
#
# FlashInfer trtllm-gen MoE module — pure tvm-ffi consumer.
#
# Like the FMHA submodule, MoE does NOT compile or vendor FlashInfer source.
# We dlopen the .so that flashinfer-python emits via its JIT at first use of
# `trtllm_*_moe`, and call into it via tvm-ffi.  This keeps lmdeploy out of
# the trtllm-gen build chain and leaves the runtime path overrideable.
#
# Docker builds run `docker/prepare_flashinfer_artifacts.sh` before CMake
# configure.  It warms the FlashInfer JIT cache and writes the resolved paths
# into `lmdeploy_artifacts.cmake`; this file only consumes those paths and
# bakes them into `trtllm_fused_moe_config.h`.  If no prepared path is found,
# we bake an empty path and require the runtime override
# `LMDEPLOY_FLASHINFER_MOE_SO=<path>`.
#
# Submodule activation is gated on CUDA >= 12.8.  Runtime selection is via
# `EngineParam::moe_backend`, checked by `MoeFfnLayer` at construction.

cmake_minimum_required(VERSION 3.24)

if(NOT (DEFINED CMAKE_CUDA_COMPILER_VERSION
        AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "12.8"))
    message(FATAL_ERROR
        "trtllm_fused_moe_wrapper is required by models and needs CUDA >= 12.8; "
        "got CMAKE_CUDA_COMPILER_VERSION='${CMAKE_CUDA_COMPILER_VERSION}'.")
endif()

find_package(CUDAToolkit REQUIRED)
find_package(Python COMPONENTS Interpreter REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m tvm_ffi.config --cmakedir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE tvm_ffi_ROOT
  RESULT_VARIABLE _tvm_result
)
if(NOT _tvm_result EQUAL 0)
    message(FATAL_ERROR
        "apache-tvm-ffi not found. Install with `pip install apache-tvm-ffi` "
        "before building FlashInfer MoE.")
endif()
find_package(tvm_ffi CONFIG REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m tvm_ffi.config --libdir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE TVM_FFI_LIBDIR
)

# ---------------------------------------------------------------------------
# Resolve the prebuilt MoE .so path.  Docker builds prepare the JIT artifacts
# before CMake configure and pass paths through lmdeploy_artifacts.cmake.
#
# Knobs:
#   LMDEPLOY_FLASHINFER_MOE_SO_PATH (cache override)
#       Skip the Python invocation entirely; use this absolute path.
#   LMDEPLOY_FLASHINFER_CUTLASS_MOE_SO_PATH (cache override)
#       Optional cutlass fused-MoE .so used by W4A8_NVFP4_FP8.
# ---------------------------------------------------------------------------
if(NOT LMDEPLOY_FLASHINFER_MOE_SO_PATH)
    set(LMDEPLOY_FLASHINFER_MOE_SO_PATH ""
        CACHE FILEPATH "Absolute path to the FlashInfer trtllm-gen MoE .so")
    message(STATUS
        "[flashinfer-moe] no prepared MoE .so path; run "
        "`docker/prepare_flashinfer_artifacts.sh` or set "
        "LMDEPLOY_FLASHINFER_MOE_SO_PATH/LMDEPLOY_FLASHINFER_MOE_SO.")
else()
    message(STATUS "[flashinfer-moe] using prepared .so: "
                   "${LMDEPLOY_FLASHINFER_MOE_SO_PATH}")
endif()

if(NOT LMDEPLOY_FLASHINFER_CUTLASS_MOE_SO_PATH)
    set(LMDEPLOY_FLASHINFER_CUTLASS_MOE_SO_PATH ""
        CACHE FILEPATH "Absolute path to the FlashInfer CUTLASS fused-MoE .so")
    message(STATUS
        "[flashinfer-moe] no prepared CUTLASS MoE .so path; run "
        "`docker/prepare_flashinfer_artifacts.sh` or set "
        "LMDEPLOY_FLASHINFER_CUTLASS_MOE_SO_PATH/LMDEPLOY_FLASHINFER_CUTLASS_MOE_SO.")
else()
    message(STATUS "[flashinfer-moe] using prepared CUTLASS .so: "
                   "${LMDEPLOY_FLASHINFER_CUTLASS_MOE_SO_PATH}")
endif()

# Bake the configure-time-resolved path into the wrapper's header.  At
# runtime, the wrapper still consults `LMDEPLOY_FLASHINFER_MOE_SO` env
# var first, so operators can swap .so versions without a rebuild.
file(GENERATE
    OUTPUT  "${CMAKE_CURRENT_BINARY_DIR}/include/lmdeploy/trtllm_fused_moe_config.h"
    CONTENT "#pragma once\n#define LMDEPLOY_FLASHINFER_MOE_SO_PATH \"${LMDEPLOY_FLASHINFER_MOE_SO_PATH}\"\n#define LMDEPLOY_FLASHINFER_CUTLASS_MOE_SO_PATH \"${LMDEPLOY_FLASHINFER_CUTLASS_MOE_SO_PATH}\"\n"
)

# ---------------------------------------------------------------------------
# MoE static library: lmdeploy raw pointers <-> tvm::ffi::Tensor calls,
# plus the local W4A8/CUTLASS packing and dispatch helpers used by the
# FlashInfer CUTLASS fused-MoE path.
# ---------------------------------------------------------------------------
add_library(trtllm_fused_moe_wrapper STATIC
    trtllm_fused_moe_wrapper.cc
    trtllm_fused_moe_helper.cu
)
target_include_directories(trtllm_fused_moe_wrapper PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}/include
    ${CMAKE_SOURCE_DIR}
)
tvm_ffi_configure_target(trtllm_fused_moe_wrapper)
target_link_libraries(trtllm_fused_moe_wrapper PUBLIC
    CUDA::cudart
    core
    quantization_kernels
    cuda_utils
    logger
)
set_target_properties(trtllm_fused_moe_wrapper PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# ---------------------------------------------------------------------------
# Smoke test: dlopens the MoE .so, verifies the FFI entries are callable.
# Doesn't run a real kernel (that needs a B200 + valid weight tensors).
# ---------------------------------------------------------------------------
if(BUILD_TEST)
    add_executable(trtllm_fused_moe_smoke tests/test_smoke.cc)
    target_link_libraries(trtllm_fused_moe_smoke PRIVATE
        trtllm_fused_moe_wrapper
        CUDA::cudart
    )
    set_target_properties(trtllm_fused_moe_smoke PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        BUILD_RPATH "${TVM_FFI_LIBDIR};${CMAKE_BINARY_DIR}/lib"
    )
endif()
