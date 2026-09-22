#include "tree-draft-gguf-mmap-plan.h"

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
    str(b, key); u32(b, GGUF_TYPE_UINT16); u16(b, v);
}

static void kv_i32(std::vector<unsigned char> & b, const char * key, int32_t v) {
    str(b, key); u32(b, GGUF_TYPE_INT32); u32(b, static_cast<uint32_t>(v));
}

static void kv_string(std::vector<unsigned char> & b, const char * key, const std::string & v) {
    str(b, key); u32(b, GGUF_TYPE_STRING); str(b, v);
}

static void tensor(std::vector<unsigned char> & b, const std::string & name, uint64_t offset) {
    str(b, name);
    u32(b, 1);
    u64(b, 4);
    u32(b, GGML_TYPE_F32);
    u64(b, offset);
}

static std::vector<unsigned char> shard_bytes(
        uint16_t split_no,
        uint16_t split_count,
        int32_t total_tensors,
        const std::vector<std::pair<std::string, uint64_t>> & tensors,
        size_t payload_bytes,
        bool pad_file_to_64 = true) {
    std::vector<unsigned char> b(GGUF_MAGIC, GGUF_MAGIC + 4);
    u32(b, GGUF_VERSION);
    u64(b, tensors.size());
    u64(b, 5);
    kv_u16(b, "split.no", split_no);
    kv_u16(b, "split.count", split_count);
    kv_i32(b, "split.tensors.count", total_tensors);
    kv_string(b, "general.architecture", "gemma4");
    kv_string(b, "general.name", "mmap-plan-model");
    for (const auto & t : tensors) tensor(b, t.first, t.second);
    while (b.size() % 32 != 0) b.push_back(0);
    b.resize(b.size() + payload_bytes, 0);
    if (pad_file_to_64) while (b.size() % 64 != 0) b.push_back(0);
    return b;
}

static void write_file(const fs::path & p, const std::vector<unsigned char> & bytes) {
    std::ofstream f(p, std::ios::binary);
    assert(f);
    f.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    assert(f);
}

static common_tree_draft_model_source source_for(const std::vector<fs::path> & paths) {
    std::vector<std::string> inputs;
    for (const auto & p : paths) inputs.push_back(p.string());
    const auto source = common_tree_draft_model_source_probe(inputs);
    assert(source.format == COMMON_TREE_DRAFT_MODEL_FORMAT_GGUF);
    return source;
}

int main() {
    const fs::path root = fs::temp_directory_path() / "llama-tree-draft-gguf-mmap-plan-test";
    std::error_code ec;
    fs::remove_all(root, ec);
    assert(fs::create_directories(root));
    const fs::path s0 = root / "model-00001-of-00002.gguf";
    const fs::path s1 = root / "model-00002-of-00002.gguf";

    write_file(s0, shard_bytes(0, 2, 3, { { "a", 0 }, { "b", 128 } }, 256));
    write_file(s1, shard_bytes(1, 2, 3, { { "c", 0 } }, 128));
    const auto source = source_for({ s1, s0 });

    common_tree_draft_gguf_mmap_policy no_merge;
    no_merge.page_size = 64;
    no_merge.coalesce = false;
    const auto separate = common_tree_draft_gguf_plan_mmap_regions(source, no_merge);
    assert(separate);
    assert(separate.plan.regions.size() == 3);
    for (const auto & region : separate.plan.regions) {
        assert(region.file_offset % 64 == 0);
        assert(region.length % 64 == 0);
        assert(region.tensors.size() == 1);
        assert(region.tensors[0].intra_map_offset == region.tensors[0].file_offset - region.file_offset);
    }
    assert(separate.plan.regions[0].shard_index == 0);
    assert(separate.plan.regions[1].shard_index == 0);
    assert(separate.plan.regions[2].shard_index == 1);

    common_tree_draft_gguf_mmap_policy merge;
    merge.page_size = 64;
    merge.max_gap = 64;
    merge.max_region_size = 256;
    const auto merged = common_tree_draft_gguf_plan_mmap_regions(source, merge);
    assert(merged);
    assert(merged.plan.regions.size() == 2);
    assert(merged.plan.regions[0].shard_index == 0);
    assert(merged.plan.regions[0].tensors.size() == 2);
    assert(merged.plan.regions[1].shard_index == 1);
    assert(merged.plan.regions[1].tensors.size() == 1);

    common_tree_draft_gguf_mmap_policy bounded = merge;
    bounded.max_region_size = 128;
    const auto bounded_plan = common_tree_draft_gguf_plan_mmap_regions(source, bounded);
    assert(bounded_plan);
    assert(bounded_plan.plan.regions.size() == 3);

    common_tree_draft_gguf_mmap_policy invalid;
    invalid.page_size = 0;
    assert(common_tree_draft_gguf_plan_mmap_regions(source, invalid).error == COMMON_TREE_DRAFT_GGUF_MMAP_PLAN_INVALID_POLICY);

    const fs::path short0 = root / "short-00001-of-00001.gguf";
    auto short_bytes = shard_bytes(0, 1, 1, { { "tail", 0 } }, 16, false);
    assert(short_bytes.size() % 64 != 0);
    write_file(short0, short_bytes);
    const auto short_source = source_for({ short0 });
    common_tree_draft_gguf_mmap_policy page64;
    page64.page_size = 64;
    const auto short_result = common_tree_draft_gguf_plan_mmap_regions(short_source, page64);
    assert(!short_result);
    assert(short_result.error == COMMON_TREE_DRAFT_GGUF_MMAP_PLAN_OUT_OF_BOUNDS);

    fs::remove_all(root, ec);
    return 0;
}
