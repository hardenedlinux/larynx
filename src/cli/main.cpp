// larynx CLI entry point. Phase 1 skeleton: the pipeline stages are stubs.
// The DSP frontend is implemented and validated separately (tests/verify_dsp.py
// drives the `larynx_dump` utility); no end-to-end synthesis yet.

#include <cstdio>

#include "larynx/dsp/fbank.h"
#include "larynx/dsp/log_mel.h"
#include "larynx/flow/flow.h"
#include "larynx/frontend/frontend.h"
#include "larynx/hift/hift.h"
#include "larynx/llm/llm.h"
#include "larynx/pipeline/pipeline.h"

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;

  std::printf("larynx 0.1.0 — Phase 1 skeleton\n");
  std::printf("  dsp      : %s\n", "log-mel (128) + fbank (80), implemented");
  std::printf("  %s\n", larynx::frontend::module_name().c_str());
  std::printf("  %s\n", larynx::llm::module_name().c_str());
  std::printf("  %s\n", larynx::flow::module_name().c_str());
  std::printf("  %s\n", larynx::hift::module_name().c_str());
  std::printf("  %s\n", larynx::pipeline::module_name().c_str());
  std::printf("No end-to-end synthesis in this phase.\n");
  return 0;
}
