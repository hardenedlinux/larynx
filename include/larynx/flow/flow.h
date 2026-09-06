#pragma once

#include <string>

namespace larynx::flow {

// Flow decoder (GGML + CUDA): PreLookaheadLayer + 22-block DiT + CFM Euler
// solver. Not implemented in Phase 1.
std::string module_name();

}  // namespace larynx::flow
