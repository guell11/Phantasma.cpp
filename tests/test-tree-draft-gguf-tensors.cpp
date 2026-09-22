#include "tree-draft-gguf-tensors.h"

#include "gguf.h"

#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static void u32(std::vector<unsigned char> & b, uint32_t v) {
    for (int i = 0; i < 4; ++i) b.push_back(static_cast<unsigned char>((v >> (8*i)) & 0xff));
}

static void u64(std::vector<unsigned char> & b, uint64_t v) {
    for (int i = 0; i < 8; ++i) b.push_back(static_cast<unsigned char>((v >> (8*i)) & 0xff));
}

static void str(std::vector<unsigned char> & b, const std::string & s) {
    u64(b, s.size());
    b.insert(b.end(), s.begin(), s.end());
}

static std::vector<unsigned char> header(uint64_t n_tensors, uint64_t n_kv = 0) {
    std::vector<unsigned char> b(GGUF_MAGIC, GGUF_MAGIC + 4);
    u32(b, GGUF_VERSION);
    u64(b, n_tensors);
    u64(b, n_kv);
    return b;
}

static void tensor(std::vector<unsigned char> & b, const std::string & name, const std::vector<uint64_t> & dims, uint32_t type, uint64_t offset) {
    str(b, name);
    u32(b, static_cast<uint32_t>(dims.size()));
    for (uint64_t d : dims) u64(b, d);
    u32(b, type);
    u64(b, offset);
}

static void pad32(std::vector<unsigned char> & b) {
    while (b.size() % 32 != 0) b.push_back(0);
}

static void write_file(const fs::path & p, const std::vector<unsigned char> & b) {
    std::ofstream f(p, std::ios::binary);
    assert(f);
    f.write(reinterpret_cast<const char *>(b.data()), static_cast<std::streamsize>(b.size()));
    assert(f);
}

static common_tree_draft_gguf_tensor_result parse(const fs::path & p, const std::vector<unsigned char> & b) {
    write_file(p, b);
    const auto source = common_tree_draft_model_source_probe(p.string());
    assert(source.format == COMMON_TREE_DRAFT_MODEL_FORMAT_GGUF);
    return common_tree_draft_gguf_tensor_directory_parse(source);
}

int main() {
    const fs::path root = fs::temp_directory_path() / "llama-tree-draft-gguf-tensors-test";
    std::error_code ec;
    fs::remove_all(root, ec);
    assert(fs::create_directories(root));
    const fs::path p = root / "model.gguf";

    auto valid = header(2);
    tensor(valid, "a", { 4, 2 }, GGML_TYPE_F32, 0);
    tensor(valid, "b", { 8 }, GGML_TYPE_F32, 32);
    const size_t directory_end = valid.size();
    pad32(valid);
    const size_t data_base = valid.size();
    valid.resize(data_base + 64, 0);
    const auto ok = parse(p, valid);
    assert(ok);
    assert(ok.directory.tensors.size() == 2);
    assert(ok.directory.directory_end == directory_end);
    assert(ok.directory.data_base == data_base);
    assert(ok.directory.tensors[0].storage_size == 32);
    assert(ok.directory.tensors[1].storage_size == 32);

    auto duplicate = header(2);
    tensor(duplicate, "dup", { 1 }, GGML_TYPE_F32, 0);
    tensor(duplicate, "dup", { 1 }, GGML_TYPE_F32, 4);
    pad32(duplicate);
    duplicate.resize(duplicate.size() + 8, 0);
    assert(parse(p, duplicate).error == COMMON_TREE_DRAFT_GGUF_TENSOR_DUPLICATE_NAME);

    auto rank0 = header(1);
    tensor(rank0, "bad", {}, GGML_TYPE_F32, 0);
    pad32(rank0);
    rank0.resize(rank0.size() + 4, 0);
    assert(parse(p, rank0).error == COMMON_TREE_DRAFT_GGUF_TENSOR_INVALID_RANK);

    auto dim0 = header(1);
    tensor(dim0, "bad", { 0 }, GGML_TYPE_F32, 0);
    pad32(dim0);
    assert(parse(p, dim0).error == COMMON_TREE_DRAFT_GGUF_TENSOR_INVALID_DIMENSION);

    auto bad_type = header(1);
    tensor(bad_type, "bad", { 4 }, GGML_TYPE_COUNT, 0);
    pad32(bad_type);
    assert(parse(p, bad_type).error == COMMON_TREE_DRAFT_GGUF_TENSOR_INVALID_TYPE);

    auto overflow = header(1);
    tensor(overflow, "huge", { static_cast<uint64_t>(INT64_MAX), 3 }, GGML_TYPE_F32, 0);
    pad32(overflow);
    assert(parse(p, overflow).error == COMMON_TREE_DRAFT_GGUF_TENSOR_SIZE_OVERFLOW);

    auto out = header(1);
    tensor(out, "outside", { 4 }, GGML_TYPE_F32, UINT64_MAX - 7);
    pad32(out);
    out.resize(out.size() + 16, 0);
    assert(parse(p, out).error == COMMON_TREE_DRAFT_GGUF_TENSOR_SPAN_OUT_OF_BOUNDS);

    auto truncated = header(1);
    str(truncated, "cut");
    u32(truncated, 1);
    assert(parse(p, truncated).error == COMMON_TREE_DRAFT_GGUF_TENSOR_TRUNCATED);

    fs::remove_all(root, ec);
    return 0;
}
