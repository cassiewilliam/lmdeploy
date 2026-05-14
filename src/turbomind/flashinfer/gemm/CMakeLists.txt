# Copyright (c) OpenMMLab. All rights reserved.

# FlashInfer SM10x dense GEMM bridge.  Docker builds prepare the
# flashinfer-python/JIT-produced .so files before CMake configure; this target
# only bakes resolved paths into a generated header and keeps runtime env-var
# overrides available.

cmake_minimum_required(VERSION 3.24)

if(NOT (DEFINED CMAKE_CUDA_COMPILER_VERSION
        AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "12.8"))
    message(FATAL_ERROR
        "flashinfer_gemm_wrapper is required by models and needs CUDA >= 12.8; "
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
        "before building FlashInfer GEMM.")
endif()
find_package(tvm_ffi CONFIG REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m tvm_ffi.config --libdir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE TVM_FFI_LIBDIR
)

if(NOT LMDEPLOY_FLASHINFER_GEMM_SO_PATH)
    set(LMDEPLOY_FLASHINFER_GEMM_SO_PATH ""
        CACHE FILEPATH "Absolute path to the FlashInfer SM10x GEMM .so")
    message(STATUS
        "[flashinfer-gemm] no prepared GEMM .so path; run "
        "`docker/prepare_flashinfer_artifacts.sh` or set "
        "LMDEPLOY_FLASHINFER_GEMM_SO_PATH/LMDEPLOY_FLASHINFER_GEMM_SO.")
else()
    message(STATUS "[flashinfer-gemm] using prepared .so: ${LMDEPLOY_FLASHINFER_GEMM_SO_PATH}")
endif()

if(NOT LMDEPLOY_FLASHINFER_FP4_GEMM_SO_PATH)
    set(LMDEPLOY_FLASHINFER_FP4_GEMM_SO_PATH ""
        CACHE FILEPATH "Absolute path to the FlashInfer SM10x FP4 GEMM .so")
    message(STATUS
        "[flashinfer-gemm] no prepared FP4 GEMM .so path; run "
        "`docker/prepare_flashinfer_artifacts.sh` or set "
        "LMDEPLOY_FLASHINFER_FP4_GEMM_SO_PATH/LMDEPLOY_FLASHINFER_FP4_GEMM_SO.")
else()
    message(STATUS "[flashinfer-gemm] using prepared FP4 .so: ${LMDEPLOY_FLASHINFER_FP4_GEMM_SO_PATH}")
endif()

if(NOT LMDEPLOY_FLASHINFER_FP4_GEMM_SM100_SO_PATH)
    set(LMDEPLOY_FLASHINFER_FP4_GEMM_SM100_SO_PATH "${LMDEPLOY_FLASHINFER_FP4_GEMM_SO_PATH}"
        CACHE FILEPATH "Absolute path to the FlashInfer SM100 FP4 GEMM .so")
endif()
if(NOT LMDEPLOY_FLASHINFER_FP4_GEMM_SM103_SO_PATH)
    set(LMDEPLOY_FLASHINFER_FP4_GEMM_SM103_SO_PATH ""
        CACHE FILEPATH "Absolute path to the FlashInfer SM103 FP4 GEMM .so")
endif()

file(GENERATE
    OUTPUT  "${CMAKE_CURRENT_BINARY_DIR}/include/lmdeploy/flashinfer_gemm_config.h"
    CONTENT "#pragma once\n#define LMDEPLOY_FLASHINFER_GEMM_SO_PATH \"${LMDEPLOY_FLASHINFER_GEMM_SO_PATH}\"\n#define LMDEPLOY_FLASHINFER_FP4_GEMM_SO_PATH \"${LMDEPLOY_FLASHINFER_FP4_GEMM_SO_PATH}\"\n#define LMDEPLOY_FLASHINFER_FP4_GEMM_SM100_SO_PATH \"${LMDEPLOY_FLASHINFER_FP4_GEMM_SM100_SO_PATH}\"\n#define LMDEPLOY_FLASHINFER_FP4_GEMM_SM103_SO_PATH \"${LMDEPLOY_FLASHINFER_FP4_GEMM_SM103_SO_PATH}\"\n"
)

add_library(flashinfer_gemm_wrapper STATIC
    flashinfer_gemm_wrapper.cc
    flashinfer_gemm_quant.cu
)
target_include_directories(flashinfer_gemm_wrapper PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}/include
    ${CMAKE_SOURCE_DIR}
)
tvm_ffi_configure_target(flashinfer_gemm_wrapper)
target_link_libraries(flashinfer_gemm_wrapper PUBLIC
    CUDA::cudart
    CUDA::cublas
    CUDA::cublasLt
    core
    gemm2
    gpt_kernels
    logger
)
set_target_properties(flashinfer_gemm_wrapper PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

if(TARGET test_gemm_v2)
    target_link_libraries(test_gemm_v2 PRIVATE flashinfer_gemm_wrapper)
endif()

if(BUILD_TEST)
    add_executable(flashinfer_gemm_smoke tests/test_smoke.cu)
    target_link_libraries(flashinfer_gemm_smoke PRIVATE
        flashinfer_gemm_wrapper
        CUDA::cudart
    )
    set_target_properties(flashinfer_gemm_smoke PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        BUILD_RPATH "${TVM_FFI_LIBDIR};${CMAKE_BINARY_DIR}/lib"
    )
endif()
