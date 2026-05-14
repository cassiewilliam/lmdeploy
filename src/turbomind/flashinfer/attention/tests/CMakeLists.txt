# Copyright (c) OpenMMLab. All rights reserved.

cmake_minimum_required(VERSION 3.26)
project(test_trtllm_attention)

find_package(CUDAToolkit REQUIRED)

# FMHA tests share the same support sources + link set.
set(_FMHA_TEST_SUPPORT_SRCS
    ${CMAKE_SOURCE_DIR}/src/turbomind/kernels/attention/test_utils.cu
    ${CMAKE_SOURCE_DIR}/src/turbomind/kernels/attention/reference.cu
)

set(_FMHA_TEST_LIBS
    trtllm_attention
    attention
    models
    unfused_attention_kernels
    core
    cuda_utils
    logger
    CUDA::cudart
    CUDA::cublas
    CUDA::curand
    nvidia::cutlass::cutlass
)

function(_lmdeploy_add_fmha_test target src)
    add_executable(${target} ${src} ${_FMHA_TEST_SUPPORT_SRCS})
    set_property(TARGET ${target} PROPERTY CUDA_SEPARABLE_COMPILATION OFF)
    set_property(TARGET ${target} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
    target_link_libraries(${target} PRIVATE ${_FMHA_TEST_LIBS})
    target_include_directories(${target} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_SOURCE_DIR}/src
    )
    target_compile_features(${target} PRIVATE cxx_std_17)
    set_target_properties(${target} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/"
    )
endfunction()

_lmdeploy_add_fmha_test(test_dc_fmha test_dc_fmha.cu)
_lmdeploy_add_fmha_test(test_ap_fmha test_ap_fmha.cu)
_lmdeploy_add_fmha_test(test_fp4_kv_cache test_fp4_kv_cache.cu)

install(TARGETS test_dc_fmha test_fp4_kv_cache
    RUNTIME DESTINATION bin/
    COMPONENT tests
)
