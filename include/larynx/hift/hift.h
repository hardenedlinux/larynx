#pragma once

#include <string>

namespace larynx::hift {

// HiFT vocoder (GGML + CUDA): upsampling convs, F0 predictor, ISTFT via
// DFT-matrix matmul + overlap-add. Not implemented in Phase 1.
std::string module_name();

}  // namespace larynx::hift
