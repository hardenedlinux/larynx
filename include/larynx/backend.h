#pragma once

// Minimal GGML CUDA-backend wiring shared by the Flow decoder and HiFT vocoder.
//
// "Registering" the CUDA backend needs no explicit call: the ggml backend
// registry is populated automatically at static-init time when GGML_USE_CUDA is
// defined (i.e. GGML_CUDA=ON at CMake time). We simply pick it up via
// ggml_backend_init_best(). This header also owns the one piece of plumbing a
// GPU backend adds over the CPU path: model weights (loaded by gguf into a host
// context) must be uploaded into a device buffer, and every tensor read/write
// must go through ggml_backend_tensor_get/set instead of a raw memcpy.

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace larynx {

// Init the best available compute backend (CUDA/GPU when compiled in and a
// device is present, else CPU). Logs the choice. Returns nullptr on failure.
// Set LARYNX_BACKEND=cpu to force the CPU backend (used by the numerical
// verify scripts so they don't depend on an idle GPU).
inline ggml_backend_t backend_init_best() {
  const char* force = std::getenv("LARYNX_BACKEND");
  ggml_backend_t b = nullptr;
  if (force && std::strcmp(force, "cpu") == 0) {
    b = ggml_backend_cpu_init();
  } else {
    b = ggml_backend_init_best();
  }
  if (!b) b = ggml_backend_cpu_init();
  if (b) {
    fprintf(stderr, "larynx: ggml backend = %s\n", ggml_backend_name(b));
  }
  return b;
}

// Upload every tensor in `wctx` (which must have been created by
// gguf_init_from_file with no_alloc=true) into a `backend` buffer, reading the
// raw bytes straight out of the gguf file. Returns the buffer (owned by the
// caller; free with ggml_backend_buffer_free). Returns nullptr on failure.
inline ggml_backend_buffer_t upload_gguf_weights(gguf_context* gctx,
                                                 ggml_context* wctx,
                                                 ggml_backend_t backend,
                                                 const std::string& path) {
  ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(wctx, backend);
  if (!buf) {
    fprintf(stderr, "larynx: ggml_backend_alloc_ctx_tensors returned null\n");
    return nullptr;
  }

  FILE* f = fopen(path.c_str(), "rb");
  if (!f) {
    fprintf(stderr, "larynx: cannot reopen '%s' to upload weights\n", path.c_str());
    return buf;  // buffer is still valid; caller frees it
  }

  const size_t data_off = gguf_get_data_offset(gctx);
  std::vector<char> tmp;
  bool ok = true;
  for (int64_t i = 0; i < gguf_get_n_tensors(gctx); i++) {
    const char* name = gguf_get_tensor_name(gctx, i);
    ggml_tensor* t = ggml_get_tensor(wctx, name);
    if (!t) continue;  // tensor present in the file but not referenced here
    const size_t nbytes = ggml_nbytes(t);
    tmp.resize(nbytes);
    const size_t off = data_off + gguf_get_tensor_offset(gctx, i);
    if (fseek(f, (long)off, SEEK_SET) != 0 || fread(tmp.data(), 1, nbytes, f) != nbytes) {
      fprintf(stderr, "larynx: failed to read tensor '%s' from '%s'\n", name, path.c_str());
      ok = false;
      break;
    }
    ggml_backend_tensor_set(t, tmp.data(), 0, nbytes);
  }
  fclose(f);
  if (!ok) {
    ggml_backend_buffer_free(buf);
    return nullptr;
  }
  return buf;
}

// Collects host-side data writes for constant / zero tensors created during a
// no_alloc graph build. Under no_alloc=true, ggml_new_tensor_* leaves data ==
// NULL, so constants must be written *after* ggml_backend_alloc_ctx_tensors has
// placed the tensors in a backend buffer. Call apply() after that alloc and
// before ggml_backend_graph_compute.
struct TensorInit {
  struct Fill {
    ggml_tensor* t;
    std::vector<float> data;
  };
  std::vector<Fill> fills;         // set to the given host data
  std::vector<ggml_tensor*> zeros; // memset to 0

  void fill(ggml_tensor* t, const std::vector<float>& data) { fills.push_back({t, data}); }
  void fill_scalar(ggml_tensor* t, float v) { fills.push_back({t, {v}}); }
  void zero(ggml_tensor* t) { zeros.push_back(t); }

  void apply() {
    for (const auto& f : fills) {
      ggml_backend_tensor_set(f.t, f.data.data(), 0, f.data.size() * sizeof(float));
    }
    for (auto* z : zeros) {
      ggml_backend_tensor_memset(z, 0, 0, ggml_nbytes(z));
    }
    fills.clear();
    zeros.clear();
  }
};

}  // namespace larynx
