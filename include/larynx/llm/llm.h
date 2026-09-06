#pragma once

#include <string>

namespace larynx::llm {

// Qwen2 backbone (GGML + CUDA) with the custom speech_embedding / llm_decoder
// heads and reimplemented ras_sampling. Not implemented in Phase 1.
std::string module_name();

}  // namespace larynx::llm
