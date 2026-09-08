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
#
# GGML is statically linked into larynx and its tools so the built artifacts are
# self-contained: BUILD_SHARED_LIBS=OFF makes ggml-base/ggml-cpu/ggml-cuda static
# archives instead of shared libraries (ldd must not show libggml-*.so). This is
# distinct from GGML_STATIC, which adds a full `-static` system link and is NOT
# what we want here. See docs/adr/0002.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "build ggml as static libraries" FORCE)

set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
if(LARYNX_ENABLE_CUDA AND CMAKE_CUDA_COMPILER)
  set(GGML_CUDA ON CACHE BOOL "" FORCE)
endif()

# --- Apply vendored-GGML patches (idempotent) --------------------------------
# GGML is pinned as a git submodule, but we carry local source patches that fix
# upstream behavior we cannot accept as-is (see docs/adr/0002). Patches are
# applied here at configure time, before ggml compiles. Idempotent: a patch whose
# change is already present is skipped; a patch that applies neither forward nor
# reverse (upstream diverged) aborts configure with a clear message rather than
# silently building wrong code.
set(LARYNX_GGML_SRC "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml")
file(GLOB LARYNX_GGML_PATCHES "${CMAKE_CURRENT_SOURCE_DIR}/patches/*.patch")
list(SORT LARYNX_GGML_PATCHES)
foreach(_patch IN LISTS LARYNX_GGML_PATCHES)
  get_filename_component(_patch_name "${_patch}" NAME)

  # Forward check: would this patch apply onto the current tree?
  execute_process(
    COMMAND git apply --check "${_patch}"
    WORKING_DIRECTORY "${LARYNX_GGML_SRC}"
    RESULT_VARIABLE _fwd
    OUTPUT_VARIABLE _fwd_out ERROR_VARIABLE _fwd_err)
  if(_fwd EQUAL 0)
    execute_process(
      COMMAND git apply "${_patch}"
      WORKING_DIRECTORY "${LARYNX_GGML_SRC}"
      RESULT_VARIABLE _apply
      OUTPUT_VARIABLE _apply_out ERROR_VARIABLE _apply_err)
    if(NOT _apply EQUAL 0)
      message(FATAL_ERROR "Failed to apply ggml patch ${_patch_name}:\n${_apply_err}")
    endif()
    message(STATUS "Applied ggml patch: ${_patch_name}")
    continue()
  endif()

  # Reverse check: is the change already present?
  execute_process(
    COMMAND git apply --check --reverse "${_patch}"
    WORKING_DIRECTORY "${LARYNX_GGML_SRC}"
    RESULT_VARIABLE _rev
    OUTPUT_VARIABLE _rev_out ERROR_VARIABLE _rev_err)
  if(_rev EQUAL 0)
    message(STATUS "ggml patch already applied (skipping): ${_patch_name}")
    continue()
  endif()

  # Neither applies -> upstream source changed; do not build silently.
  message(FATAL_ERROR
    "patch 应用失败，可能是 GGML 版本升级导致代码变化，需要手动检查更新 patch 内容: ${_patch_name}\n"
    "  forward: ${_fwd_err}\n  reverse: ${_rev_err}")
endforeach()

add_subdirectory(third_party/ggml)

# --- ICU (Unicode NFC + character properties for the text tokenizer) ---------
# The Qwen2Tokenizer must be bit-exact with the Rust tokenizers crate, so we use
# ICU's UCD tables for NFC (unorm2) and \p{L}/\p{N}/\s (uchar) instead of
# hand-rolled Unicode data.
find_package(ICU COMPONENTS uc i18n REQUIRED)

# --- DSP frontend (pure C++, no dependencies) --------------------------------
add_library(larynx_dsp STATIC
  src/dsp/log_mel.cpp
  src/dsp/fbank.cpp
)
target_include_directories(larynx_dsp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# --- Placeholder stage modules (shells, to be implemented in later phases) ---
add_library(larynx_frontend STATIC src/frontend/frontend.cpp)
target_include_directories(larynx_frontend PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(larynx_llm STATIC
  src/llm/llm.cpp
  src/llm/ops.cpp
  src/llm/gguf.cpp
  src/llm/tokenizer.cpp
  src/llm/sampling.cpp
)
target_include_directories(larynx_llm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(larynx_llm PUBLIC ICU::uc ICU::i18n ggml)

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

add_library(larynx_hift STATIC
  src/hift/hift.cpp
  src/hift/ops.cpp
  src/hift/gguf.cpp
  src/hift/source.cpp
)
target_include_directories(larynx_hift PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(larynx_hift PUBLIC ggml)

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

# Test utility used by tests/verify_hift.py: runs the HiFT vocoder and dumps
# per-stage tensors for comparison against the PyTorch reference.
add_executable(larynx_hift_dump tests/hift_dump.cpp)
target_link_libraries(larynx_hift_dump PRIVATE larynx_hift)

# Test utility used by tests/verify_tokenizer.py: runs the C++ text tokenizer
# and dumps token ids for bit-exact comparison against CosyVoice3Tokenizer.
add_executable(larynx_tokenizer_dump tests/tokenizer_dump.cpp)
target_link_libraries(larynx_tokenizer_dump PRIVATE larynx_llm)

# Test utility used by tests/verify_llm.py: runs the Qwen2 prefill and dumps
# per-layer hidden states + final_norm + logits for comparison against the
# PyTorch reference.
add_executable(larynx_llm_dump tests/llm_dump.cpp)
target_link_libraries(larynx_llm_dump PRIVATE larynx_llm)

# Test utility used by tests/verify_llm_decode.py: runs prefill + one KV-cache
# decode step and dumps the cache + decode logits/final_norm.
add_executable(larynx_llm_decode_dump tests/llm_decode_dump.cpp)
target_link_libraries(larynx_llm_decode_dump PRIVATE larynx_llm)

# Test utility used by tests/verify_llm_decode_seq.py: runs prefill + the full
# autoregressive decode (injecting a captured token trajectory) and dumps the
# per-step llm_decoder logits.
add_executable(larynx_llm_decode_seq_dump tests/llm_decode_seq_dump.cpp)
target_link_libraries(larynx_llm_decode_seq_dump PRIVATE larynx_llm)

# Test utility used by tests/verify_llm_sampling.py: reproduces the deterministic
# nucleus candidate set from raw per-step logits (sampling policy, no model run).
add_executable(larynx_llm_sample_dump tests/llm_sample_dump.cpp)
target_link_libraries(larynx_llm_sample_dump PRIVATE larynx_llm)

# Test utility used by tests/verify_generate.py: runs the fully autonomous
# prefill -> sample -> KV-cache decode -> stop loop (own RNG, no injected history).
add_executable(larynx_llm_generate_dump tests/llm_generate_dump.cpp)
target_link_libraries(larynx_llm_generate_dump PRIVATE larynx_llm)

# --- Tests (Python-driven numerical verification) ----------------------------
# The verify scripts need numpy (+ torch for the flow/hift ones). Use the repo's
# .venv python rather than the system `python3`, which has neither.
set(LARYNX_PYTHON "${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin/python" CACHE FILEPATH
    "Python interpreter (with numpy/torch) used to run the verify scripts")
enable_testing()
add_test(NAME dsp_numerical
  COMMAND ${CMAKE_COMMAND} -E env
          LARYNX_DUMP=$<TARGET_FILE:larynx_dump>
          ${LARYNX_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/tests/verify_dsp.py)
add_test(NAME flow_numerical
  COMMAND ${CMAKE_COMMAND} -E env
          LARYNX_FLOW_DUMP=$<TARGET_FILE:larynx_flow_dump>
          ${LARYNX_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/tests/verify_flow.py)
add_test(NAME hift_numerical
  COMMAND ${CMAKE_COMMAND} -E env
          LARYNX_HIFT_DUMP=$<TARGET_FILE:larynx_hift_dump>
          ${LARYNX_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/tests/verify_hift.py)
add_test(NAME tokenizer_numerical
  COMMAND ${CMAKE_COMMAND} -E env
          LARYNX_TOKENIZER_DUMP=$<TARGET_FILE:larynx_tokenizer_dump>
          ${LARYNX_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/tests/verify_tokenizer.py)
add_test(NAME llm_numerical
  COMMAND ${CMAKE_COMMAND} -E env
          LARYNX_LLM_DUMP=$<TARGET_FILE:larynx_llm_dump>
          ${LARYNX_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/tests/verify_llm.py)
add_test(NAME llm_decode_numerical
  COMMAND ${CMAKE_COMMAND} -E env
          LARYNX_LLM_DECODE_DUMP=$<TARGET_FILE:larynx_llm_decode_dump>
          ${LARYNX_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/tests/verify_llm_decode.py)
add_test(NAME llm_decode_seq_numerical
  COMMAND ${CMAKE_COMMAND} -E env
          LARYNX_LLM_DECODE_SEQ_DUMP=$<TARGET_FILE:larynx_llm_decode_seq_dump>
          ${LARYNX_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/tests/verify_llm_decode_seq.py)
add_test(NAME llm_sampling_numerical
  COMMAND ${CMAKE_COMMAND} -E env
          LARYNX_LLM_SAMPLE_DUMP=$<TARGET_FILE:larynx_llm_sample_dump>
          ${LARYNX_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/tests/verify_llm_sampling.py)
add_test(NAME llm_generate
  COMMAND ${CMAKE_COMMAND} -E env
          LARYNX_LLM_GENERATE_DUMP=$<TARGET_FILE:larynx_llm_generate_dump>
          ${LARYNX_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/tests/verify_generate.py)
