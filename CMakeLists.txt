cmake_minimum_required(VERSION 3.16)
project(larynx VERSION 0.1.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(LARYNX_ENABLE_CUDA "Enable the CUDA backend (reserved for GGML kernels)" ON)

# --- CUDA (GGML backend) -----------------------------------------------------
# Phase 1 has no CUDA sources; this only reserves the language so later phases
# can drop in GGML's CUDA kernels without restructuring the build. GGML itself
# is integrated in a later phase (its own CMake subdirectory).
if(LARYNX_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    message(STATUS "CUDA compiler found: ${CMAKE_CUDA_COMPILER} "
                   "(no CUDA sources in Phase 1)")
  else()
    message(STATUS "CUDA not found; building CPU-only (Phase 1 needs no CUDA)")
  endif()
endif()

# --- GGML (tensor library backend, vendored as a git submodule) --------------
# One compute path for the Flow decoder (and later LLM/HiFT): GGML's own CUDA
# kernels. Build the CPU backend always; add CUDA when LARYNX_ENABLE_CUDA and a
# CUDA toolchain exist. See docs/ARCHITECTURE.md and docs/adr/0001.
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
if(LARYNX_ENABLE_CUDA AND CMAKE_CUDA_COMPILER)
  set(GGML_CUDA ON CACHE BOOL "" FORCE)
endif()
add_subdirectory(third_party/ggml)

# --- DSP frontend (pure C++, no dependencies) --------------------------------
add_library(larynx_dsp STATIC
  src/dsp/log_mel.cpp
  src/dsp/fbank.cpp
)
target_include_directories(larynx_dsp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# --- Placeholder stage modules (shells, to be implemented in later phases) ---
add_library(larynx_frontend STATIC src/frontend/frontend.cpp)
target_include_directories(larynx_frontend PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(larynx_llm STATIC src/llm/llm.cpp)
target_include_directories(larynx_llm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(larynx_flow STATIC
  src/flow/flow.cpp
  src/flow/ops.cpp
  src/flow/gguf.cpp
  src/flow/prelookahead.cpp
  src/flow/dit.cpp
  src/flow/flow_matching.cpp
)
target_include_directories(larynx_flow PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(larynx_flow PUBLIC ggml)

add_library(larynx_hift STATIC src/hift/hift.cpp)
target_include_directories(larynx_hift PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(larynx_pipeline STATIC src/pipeline/pipeline.cpp)
target_include_directories(larynx_pipeline PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# --- Executables -------------------------------------------------------------
add_executable(larynx src/cli/main.cpp)
target_link_libraries(larynx PRIVATE
  larynx_dsp
  larynx_frontend
  larynx_llm
  larynx_flow
  larynx_hift
  larynx_pipeline
)

# Test utility used by tests/verify_dsp.py for the numerical comparison.
add_executable(larynx_dump tests/dump_features.cpp)
target_link_libraries(larynx_dump PRIVATE larynx_dsp)

# Test utility used by tests/verify_flow.py: runs the Flow decoder and dumps
# per-stage tensors for comparison against the PyTorch reference.
add_executable(larynx_flow_dump tests/flow_dump.cpp)
target_link_libraries(larynx_flow_dump PRIVATE larynx_flow)

# --- Tests (Python-driven numerical verification) ----------------------------
enable_testing()
add_test(NAME dsp_numerical
  COMMAND ${CMAKE_COMMAND} -E env
          LARYNX_DUMP=$<TARGET_FILE:larynx_dump>
          python3 ${CMAKE_CURRENT_SOURCE_DIR}/tests/verify_dsp.py)
add_test(NAME flow_numerical
  COMMAND ${CMAKE_COMMAND} -E env
          LARYNX_FLOW_DUMP=$<TARGET_FILE:larynx_flow_dump>
          python3 ${CMAKE_CURRENT_SOURCE_DIR}/tests/verify_flow.py)
