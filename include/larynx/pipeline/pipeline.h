#pragma once

#include <string>

namespace larynx::pipeline {

// Top-level text + voice-script -> PCM/WAV orchestration. Not implemented in
// Phase 1 (wires the DSP frontend, frontend networks, LLM, Flow and HiFT).
std::string module_name();

}  // namespace larynx::pipeline
