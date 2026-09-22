#include "tree-draft-gguf-shards.h"

#include "gguf.h"

#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static void u16(std::vector<unsigned char> & b, uint16_t v) {
    b.push_back(static_cast<unsigned char>(v & 0xff));
    b.push_back(static_cast<unsigned char>((v >> 8) & 0xff));
}

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

static void kv_u16(std::vector<unsigned char> & b, const char * key, uint16_t v) {
    str(b, key);
    u32(b, GGUF_TYPE_UINT16);
    u16(b, v);
}

static void kv_i32(std::vector<unsigned char> & b, const char * key, int32_t v) {
    str(b, key);
    u32(b, GGUF_TYPE_INT32);
    u32(b, static_cast<uint32_t>(v));
}

static void kv_string(std::vector<unsigned char> & b, const char * key, const std::string & v) {
    str(b, key);
    u32(b, GGUF_TYPE_STRING);
    str(b, v);
}

static void tensor(std::vector<unsigned char> & b, const std::string & name, uint64_t offset) {
    str(b, name);
    u32(b, 1);
    u64(b, 4);
    u32(b, GGML_TYPE_F32);
    u64(b, offset);
}

static void pad32(std::vector<unsigned char> & b) {
    while (b.size() % 32 != 0) b.push_back(0);
}

static std::vector<unsigned char> shard_bytes(
        uint16_t split_no,
        uint16_t split_count,
        int32_t total_tensors,
        const std::string & identity,
        const std::vector<std::string> & names) {
    std::vector<unsigned char> b(GGUF_MAGIC, GGUF_MAGIC + 4);
    u32(b, GGUF_VERSION);
    u64(b, names.size());
    u64(b, 5);
    kv_u16(b, "split.no", split_no);
    kv_u16(b, "split.count", split_count);
    kv_i32(b, "split.tensors.count", total_tensors);
    kv_string(b, "general.architecture", "gemma4");
    kv_string(b, "general.name", identity);
    uint64_t offset = 0;
    for (const auto & name : names) {
        tensor(b, name, offset);
        offset += 16;
    }
    pad32(b);
    b.resize(b.size() + static_cast<size_t>(offset), 0);
    return b;
}

static void write_file(const fs::path & p, const std::vector<unsigned char> & b) {
    std::ofstream f(p, std::ios::binary);
    assert(f);
    f.write(reinterpret_cast<const char *>(b.data()), static_cast<std::streamsize>(b.size()));
    assert(f);
}

static common_tree_draft_gguf_shard_result resolve(const std::vector<fs::path> & paths) {
    std::vector<std::string> inputs;
    for (const auto & p : paths) inputs.push_back(p.string());
    const auto source = common_tree_draft_model_source_probe(inputs);
    assert(source.format == COMMON_TREE_DRAFT_MODEL_FORMAT_GGUF);
    assert(source.loader == COMMON_TREE_DRAFT_MODEL_LOADER_GGUF);
    return common_tree_draft_gguf_resolve_shards(source);
}

int main() {
    const fs::path root = fs::temp_directory_path() / "llama-tree-draft-gguf-shards-test";
    std::error_code ec;
    fs::remove_all(root, ec);
    assert(fs::create_directories(root));

    const fs::path a = root / "model-00001-of-00002.gguf";
    const fs::path b = root / "model-00002-of-00002.gguf";
    write_file(a, shard_bytes(0, 2, 3, "same-model", { "a", "b" }));
    write_file(b, shard_bytes(1, 2, 3, "same-model", { "c" }));

    const auto valid = resolve({ b, a });
    assert(valid);
    assert(valid.set.split_count == 2);
    assert(valid.set.expected_tensor_count == 3);
    assert(valid.set.tensors.size() == 3);
    assert(valid.set.tensors[0].descriptor.name == "a");
    assert(valid.set.tensors[0].shard_index == 0);
    assert(valid.set.tensors[2].descriptor.name == "c");
    assert(valid.set.tensors[2].shard_index == 1);
    assert(valid.set.source_paths.size() == 2);
    assert(valid.set.source_paths[0].find("00001") != std::string::npos);
    assert(valid.set.source_paths[1].find("00002") != std::string::npos);

    write_file(a, shard_bytes(0, 3, 2, "same-model", { "a" }));
    write_file(b, shard_bytes(1, 3, 2, "same-model", { "b" }));
    assert(resolve({ a, b }).error == COMMON_TREE_DRAFT_GGUF_SHARD_MISSING_INDEX);

    write_file(a, shard_bytes(0, 2, 2, "same-model", { "a" }));
    write_file(b, shard_bytes(0, 2, 2, "same-model", { "b" }));
    assert(resolve({ a, b }).error == COMMON_TREE_DRAFT_GGUF_SHARD_DUPLICATE_INDEX);

    write_file(a, shard_bytes(0, 2, 2, "same-model", { "a" }));
    write_file(b, shard_bytes(1, 3, 2, "same-model", { "b" }));
    assert(resolve({ a, b }).error == COMMON_TREE_DRAFT_GGUF_SHARD_INCONSISTENT_SPLIT_COUNT);

    write_file(a, shard_bytes(0, 2, 2, "model-a", { "a" }));
    write_file(b, shard_bytes(1, 2, 2, "model-b", { "b" }));
    assert(resolve({ a, b }).error == COMMON_TREE_DRAFT_GGUF_SHARD_FOREIGN_MODEL);

    write_file(a, shard_bytes(0, 2, 2, "same-model", { "dup" }));
    write_file(b, shard_bytes(1, 2, 2, "same-model", { "dup" }));
    assert(resolve({ a, b }).error == COMMON_TREE_DRAFT_GGUF_SHARD_DUPLICATE_TENSOR);

    write_file(a, shard_bytes(0, 2, 3, "same-model", { "a" }));
    write_file(b, shard_bytes(1, 2, 3, "same-model", { "b" }));
    assert(resolve({ a, b }).error == COMMON_TREE_DRAFT_GGUF_SHARD_TENSOR_COUNT_MISMATCH);

    fs::remove_all(root, ec);
    return 0;
}
