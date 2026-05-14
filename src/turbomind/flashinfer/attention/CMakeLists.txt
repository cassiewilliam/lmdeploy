# Copyright (c) OpenMMLab. All rights reserved.
#
# Build the lmdeploy-side wrapper around FlashInfer's trtllm-gen FMHA module.
#
# Refactored from a previous version that compiled FlashInfer source as a
# CMake target (FetchContent → libflashinfer_fmha.so).  The current design
# leaves source/wheel installation and JIT/AOT `.so` generation to
# `flashinfer-python`:
#   * `flashinfer-python` is provided by the build/runtime environment and
#     exposes the trtllm-gen FMHA FFI exports from its generated `.so`.
#   * Our wrapper resolves the path to that `.so` at runtime — primarily via
#     the `LMDEPLOY_FLASHINFER_FMHA_SO` env var (settable by the lmdeploy
#     Python entry point after triggering FlashInfer's JIT) — and loads it
#     through `tvm::ffi::Module::LoadFromFile()`.  Interaction with FlashInfer
#     is therefore purely tvm-ffi, no source-level coupling.
#
# This branch builds the FlashInfer FMHA wrapper by default: `engine` and
# `models` link `trtllm_attention` directly, so the target must always exist.

cmake_minimum_required(VERSION 3.24)

find_package(CUDAToolkit REQUIRED)
find_package(Python COMPONENTS Interpreter REQUIRED)

# ---------------------------------------------------------------------------
# tvm-ffi: from `pip install apache-tvm-ffi`.  Provides headers + libtvm_ffi.so.
# ---------------------------------------------------------------------------
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m tvm_ffi.config --cmakedir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE tvm_ffi_ROOT
  RESULT_VARIABLE _tvm_result
)
if(NOT _tvm_result EQUAL 0)
    message(FATAL_ERROR
        "apache-tvm-ffi not found.  Install with `pip install apache-tvm-ffi` "
        "before building FlashInfer FMHA.")
endif()
find_package(tvm_ffi CONFIG REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m tvm_ffi.config --libdir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE TVM_FFI_LIBDIR
)

# ---------------------------------------------------------------------------
# FlashInfer version pin.  Docker images install FlashInfer from wheels and
# prepare the JIT/AOT artifacts before CMake configure, so this CMake file no
# longer downloads them by default.
# ---------------------------------------------------------------------------
set(LMDEPLOY_FLASHINFER_TAG "0.6.10.post1"
    CACHE STRING "FlashInfer package version used to prepare artifacts")

if(NOT LMDEPLOY_FLASHINFER_TAG MATCHES "^(v|release-v)?0\\.6\\.(10(\\.post1)?|11)$")
    message(FATAL_ERROR
        "LMDEPLOY_FLASHINFER_TAG must be a supported 0.6.10/0.6.11 ref "
        "(got '${LMDEPLOY_FLASHINFER_TAG}'). If you need to bump to a newer "
        "one, also update Docker's FlashInfer wheel install and artifact warmup.")
endif()
string(REGEX REPLACE "^(v|release-v)" "" _LMDEPLOY_FLASHINFER_VER "${LMDEPLOY_FLASHINFER_TAG}")

# ---------------------------------------------------------------------------
# Cubin set + metainfo header — normally prepared by
# `docker/prepare_flashinfer_artifacts.sh`.
#
# Pinned hash for manual auto-download.  Docker artifact warmup normally
# derives this from `flashinfer.artifacts.ArtifactPath.TRTLLM_GEN_FMHA` in the
# installed wheel and passes it through `lmdeploy_artifacts.cmake`.
#
# `download_cubins.py` remains available for manual/local builds, but CMake
# configure no longer performs network downloads unless explicitly requested.
#
# Knobs:
#   LMDEPLOY_FLASHINFER_FMHA_CUBIN_HASH   (default = matches LMDEPLOY_FLASHINFER_TAG)
#   LMDEPLOY_FLASHINFER_FMHA_CUBIN_CACHE  (default = $HOME/.cache/flashinfer/<ver>/cubins)
#   LMDEPLOY_FLASHINFER_FMHA_AUTO_DOWNLOAD (default = OFF)
#   LMDEPLOY_FLASHINFER_FMHA_CUBIN_FILTER (regex; multiple via `;` semi-colon)
# ---------------------------------------------------------------------------

set(_LMDEPLOY_FLASHINFER_DEFAULT_CUBIN_HASH "134850621dbbd55ed6b0c3fa7c29b968136c05ef")

set(LMDEPLOY_FLASHINFER_FMHA_CUBIN_HASH "${_LMDEPLOY_FLASHINFER_DEFAULT_CUBIN_HASH}"
    CACHE STRING "FlashInfer trtllm-gen FMHA cubin artifact hash")

if(NOT DEFINED LMDEPLOY_FLASHINFER_FMHA_CUBIN_CACHE)
    set(LMDEPLOY_FLASHINFER_FMHA_CUBIN_CACHE
        "$ENV{HOME}/.cache/flashinfer/cubins"
        CACHE PATH "FlashInfer trtllm-gen FMHA cubin cache root")
endif()

option(LMDEPLOY_FLASHINFER_FMHA_AUTO_DOWNLOAD
    "Auto-download trtllm-gen cubins at configure time" OFF)

set(LMDEPLOY_FLASHINFER_FMHA_CUBIN_FILTER ""
    CACHE STRING "Optional regex(es) to restrict downloaded cubins; multiple via ';'")

if(LMDEPLOY_FLASHINFER_FMHA_AUTO_DOWNLOAD)
    if(NOT LMDEPLOY_FLASHINFER_FMHA_CUBIN_HASH)
        message(FATAL_ERROR
            "LMDEPLOY_FLASHINFER_FMHA_AUTO_DOWNLOAD=ON but LMDEPLOY_FLASHINFER_FMHA_CUBIN_HASH "
            "is not set for FlashInfer tag ${LMDEPLOY_FLASHINFER_TAG}.")
    endif()
    set(_LMDEPLOY_DL_ARGS
        "${CMAKE_CURRENT_SOURCE_DIR}/scripts/download_cubins.py"
        --hash "${LMDEPLOY_FLASHINFER_FMHA_CUBIN_HASH}"
        --dest "${LMDEPLOY_FLASHINFER_FMHA_CUBIN_CACHE}")
    foreach(_pat IN LISTS LMDEPLOY_FLASHINFER_FMHA_CUBIN_FILTER)
        if(_pat)
            list(APPEND _LMDEPLOY_DL_ARGS --filter "${_pat}")
        endif()
    endforeach()
    message(STATUS "[flashinfer-fmha] downloading cubins (hash=${LMDEPLOY_FLASHINFER_FMHA_CUBIN_HASH})...")
    execute_process(
        COMMAND "${Python_EXECUTABLE}" ${_LMDEPLOY_DL_ARGS} --quiet
        RESULT_VARIABLE _LMDEPLOY_DL_RC
        ERROR_VARIABLE  _LMDEPLOY_DL_ERR
    )
    if(NOT _LMDEPLOY_DL_RC EQUAL 0)
        message(WARNING
            "[flashinfer-fmha] cubin auto-download exited with code ${_LMDEPLOY_DL_RC}; "
            "build will fall back to whatever is already in the cache.\n"
            "stderr:\n${_LMDEPLOY_DL_ERR}")
    endif()

    add_custom_target(flashinfer_fmha_cubins
        COMMAND "${Python_EXECUTABLE}" ${_LMDEPLOY_DL_ARGS}
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        COMMENT "Refreshing FlashInfer trtllm-gen cubins")
endif()

# ---------------------------------------------------------------------------
# Resolve the trtllm-gen FMHA `.so`.  Prefer a path supplied by the artifact
# warmup script; otherwise try a lightweight cache scan via `resolve_so.py`.
# Bake the path into a generated header so the C++ wrapper can load it without
# runtime Python.
#
# Runtime override: `LMDEPLOY_FLASHINFER_FMHA_SO` env var still wins over
# the baked path — useful when redeploying on a host whose JIT cache lives
# at a different location than the one observed at configure time.
#
# If resolution fails, the baked path is set to the empty string.  The wrapper
# then falls back to the env var; the .so loader emits a clear error if neither
# is available.
# ---------------------------------------------------------------------------
if(LMDEPLOY_FLASHINFER_FMHA_SO_PATH)
    set(_LMDEPLOY_FLASHINFER_FMHA_SO_RESOLVED "${LMDEPLOY_FLASHINFER_FMHA_SO_PATH}")
    message(STATUS "[flashinfer-fmha] using prepared .so: ${_LMDEPLOY_FLASHINFER_FMHA_SO_RESOLVED}")
else()
    execute_process(
        COMMAND "${Python_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/scripts/resolve_so.py"
        OUTPUT_VARIABLE _LMDEPLOY_FLASHINFER_FMHA_SO_RESOLVED
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_VARIABLE  _LMDEPLOY_FLASHINFER_FMHA_SO_RESOLVE_ERR
        RESULT_VARIABLE _LMDEPLOY_FLASHINFER_FMHA_SO_RESOLVE_RC
    )
    if(NOT _LMDEPLOY_FLASHINFER_FMHA_SO_RESOLVE_RC EQUAL 0
       OR NOT _LMDEPLOY_FLASHINFER_FMHA_SO_RESOLVED)
        message(STATUS
            "[flashinfer-fmha] no prepared .so found; run "
            "`docker/prepare_flashinfer_artifacts.sh` or set "
            "LMDEPLOY_FLASHINFER_FMHA_SO_PATH/LMDEPLOY_FLASHINFER_FMHA_SO.\n"
            "stderr:\n${_LMDEPLOY_FLASHINFER_FMHA_SO_RESOLVE_ERR}")
        set(_LMDEPLOY_FLASHINFER_FMHA_SO_RESOLVED "")
    else()
        message(STATUS
            "[flashinfer-fmha] resolved cached .so: ${_LMDEPLOY_FLASHINFER_FMHA_SO_RESOLVED}")
    endif()
endif()

file(GENERATE
    OUTPUT  "${CMAKE_CURRENT_BINARY_DIR}/include/lmdeploy/flashinfer_fmha_config.h"
    CONTENT "#pragma once\n#define LMDEPLOY_FLASHINFER_FMHA_SO_PATH \"${_LMDEPLOY_FLASHINFER_FMHA_SO_RESOLVED}\"\n#define LMDEPLOY_FLASHINFER_FMHA_CUBIN_CACHE_PATH \"${LMDEPLOY_FLASHINFER_FMHA_CUBIN_CACHE}\"\n"
)

# Refresh target — re-run resolve_so.py on demand without reconfiguring.
add_custom_target(flashinfer_fmha_resolve_so
    COMMAND "${Python_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/scripts/resolve_so.py"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Re-resolving FlashInfer trtllm-gen FMHA .so path")

# ---------------------------------------------------------------------------
# Wrapper static library: lmdeploy raw pointers <-> tvm::ffi::Tensor calls.
# Loads `flashinfer-python`'s prebuilt trtllm-gen FMHA `.so` at runtime via
# tvm-ffi.  Path source priority: `LMDEPLOY_FLASHINFER_FMHA_SO` env var
# (runtime override) → `LMDEPLOY_FLASHINFER_FMHA_SO_PATH` macro baked above.
# ---------------------------------------------------------------------------
add_library(flashinfer_fmha_wrapper STATIC
    flashinfer_fmha_wrapper.cc
)
target_include_directories(flashinfer_fmha_wrapper PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}/include
    # Project root so consumers can use the project-wide
    # `#include "src/turbomind/flashinfer/..."` style.
    ${CMAKE_SOURCE_DIR}
)
tvm_ffi_configure_target(flashinfer_fmha_wrapper)
target_link_libraries(flashinfer_fmha_wrapper PUBLIC
    CUDA::cudart
)

set_target_properties(flashinfer_fmha_wrapper PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# trtllm_attention: KV-cache writers + FlashInfer paged FMHA dispatch.
# ---------------------------------------------------------------------------
if(NOT (DEFINED CMAKE_CUDA_COMPILER_VERSION
        AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "12.8"))
    message(FATAL_ERROR
        "trtllm_attention is required by engine/models and needs CUDA >= 12.8; "
        "got CMAKE_CUDA_COMPILER_VERSION='${CMAKE_CUDA_COMPILER_VERSION}'.")
endif()

add_library(trtllm_attention STATIC
    fmha_engine_helper.cc
    kv_cache_utils_v3.cu
)
set_property(TARGET trtllm_attention PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET trtllm_attention PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_include_directories(trtllm_attention PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}
)
target_compile_definitions(trtllm_attention PRIVATE ENABLE_FP4)
target_link_libraries(trtllm_attention PRIVATE nvidia::cutlass::cutlass)
target_link_libraries(trtllm_attention PUBLIC flashinfer_fmha_wrapper)

if(BUILD_TEST)
    add_subdirectory(tests)
endif()
