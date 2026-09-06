#pragma once

#include <string>

namespace larynx::frontend {

// ONNX Runtime C++ audio frontend: campplus (speaker embedding) and
// speech_tokenizer_v3. Deliberately kept on ONNX Runtime (see
// docs/adr/0001-drop-onnx-runtime-for-compute.md) and NOT implemented in
// Phase 1 — this module is a shell for the next phase.
std::string module_name();

}  // namespace larynx::frontend
