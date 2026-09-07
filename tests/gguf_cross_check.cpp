// GGUF cross-validation against the real GGML library.
//
// Links against a throwaway ggml checkout (third_party/ggml-verify, NOT
// committed) and loads a .gguf produced by tools/convert_weights.py using the
// official gguf_init_from_file() entry point. For every tensor it prints the
// name, GGML type, dimension list (ne), byte size, and a CRC-32 checksum
// (zlib/ISO-HDLC) of the raw data, so the Python driver (tests/cross_check.py)
// can compare against the source state_dict byte-for-byte via zlib.crc32.
//
// Build (one-off, against the throwaway ggml build):
//   c++ -std=c++17 -I third_party/ggml-verify/include \
//       tests/gguf_cross_check.cpp -L third_party/ggml-verify/build/src \
//       -lggml-base -o build/gguf_cross_check
//   (gguf/ggml symbols live in libggml-base.so; libggml.so is only the backend
//    dispatcher)
//
// This file is documentation-grade: it exists to prove the format, not as a
// build target of the main project.

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <vector>

#include "gguf.h"
#include "ggml.h"

// CRC-32/ISO-HDLC (the same as zlib.crc32): reflected polynomial 0xEDB88320,
// init 0xFFFFFFFF, no post-complement on the way in, final XOR 0xFFFFFFFF.
static uint32_t crc32(const uint8_t * p, size_t n) {
    static uint32_t table[256];
    static bool init = false;
    if (!init) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) {
                c = (c & 1u) ? (c >> 1) ^ 0xEDB88320u : (c >> 1);
            }
            table[i] = c;
        }
        init = true;
    }
    uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < n; ++i) {
        c = table[(c ^ p[i]) & 0xFFu] ^ (c >> 8);
    }
    return c ^ 0xFFFFFFFFu;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <file.gguf>\n", argv[0]);
        return 2;
    }

    struct gguf_init_params params = { /*no_alloc=*/false, /*ctx=*/nullptr };
    struct gguf_context * ctx = gguf_init_from_file(argv[1], params);
    if (ctx == nullptr) {
        std::fprintf(stderr, "gguf_init_from_file failed for %s\n", argv[1]);
        return 1;
    }

    // Read the whole file so we can slice out raw tensor bytes by file offset.
    std::ifstream f(argv[1], std::ios::binary);
    std::vector<uint8_t> file((std::istreambuf_iterator<char>(f)),
                              std::istreambuf_iterator<char>());
    if (file.empty()) {
        std::fprintf(stderr, "failed to read %s\n", argv[1]);
        return 1;
    }

    const size_t data_off = gguf_get_data_offset(ctx);
    const int64_t n = gguf_get_n_tensors(ctx);

    std::printf("GGUF version=%u alignment=%zu n_tensors=%lld n_kv=%lld\n",
                gguf_get_version(ctx), gguf_get_alignment(ctx),
                static_cast<long long>(n),
                static_cast<long long>(gguf_get_n_kv(ctx)));

    for (int64_t i = 0; i < n; ++i) {
        const char * name = gguf_get_tensor_name(ctx, i);
        enum ggml_type type = gguf_get_tensor_type(ctx, i);
        const int64_t * ne = gguf_get_tensor_ne(ctx, i);
        const size_t size = gguf_get_tensor_size(ctx, i);
        const size_t off = gguf_get_tensor_offset(ctx, i);

        // gguf_get_data_offset() is the file offset of the data blob;
        // gguf_get_tensor_offset() is the tensor's offset relative to the blob.
        const uint8_t * p = file.data() + data_off + off;

        std::printf(
            "TENSOR %s type=%s ne=%lld,%lld,%lld,%lld size=%zu checksum=%08x\n",
            name, ggml_type_name(type),
            static_cast<long long>(ne[0]), static_cast<long long>(ne[1]),
            static_cast<long long>(ne[2]), static_cast<long long>(ne[3]),
            size, static_cast<unsigned int>(crc32(p, size)));
    }

    gguf_free(ctx);
    return 0;
}
