#include "tree-draft-model-source.h"

#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static void write_bytes(const fs::path & path, const std::vector<unsigned char> & bytes) {
    std::ofstream file(path, std::ios::binary);
    assert(file);
    file.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    assert(file);
}

static void append_u64_le(std::vector<unsigned char> & bytes, uint64_t value) {
    for (int i = 0; i < 8; ++i) {
        bytes.push_back(static_cast<unsigned char>((value >> (8 * i)) & 0xff));
    }
}

static void write_gguf_probe_fixture(const fs::path & path) {
    std::vector<unsigned char> bytes = {
        'G', 'G', 'U', 'F',
        3, 0, 0, 0,
    };
    append_u64_le(bytes, 1);
    append_u64_le(bytes, 0);
    assert(bytes.size() == 24);
    write_bytes(path, bytes);
}

static void write_safetensors_probe_fixture(const fs::path & path, const std::string & header) {
    std::vector<unsigned char> bytes;
    append_u64_le(bytes, header.size());
    bytes.insert(bytes.end(), header.begin(), header.end());
    write_bytes(path, bytes);
}

static const common_tree_draft_model_probe_evidence * find_signal(
        const common_tree_draft_model_source & source,
        common_tree_draft_model_probe_signal signal) {
    for (const auto & item : source.evidence) {
        if (item.signal == signal) {
            return &item;
        }
    }
    return nullptr;
}

int main() {
    const fs::path root = fs::temp_directory_path() / "llama-tree-draft-model-source-test";
    std::error_code ec;
    fs::remove_all(root, ec);
    assert(fs::create_directories(root));

    const fs::path gguf = root / "model.gguf";
    write_gguf_probe_fixture(gguf);
    const auto gguf_probe = common_tree_draft_model_source_probe(gguf.string());
    assert(gguf_probe.format == COMMON_TREE_DRAFT_MODEL_FORMAT_GGUF);
    assert(gguf_probe.loader == COMMON_TREE_DRAFT_MODEL_LOADER_GGUF);
    assert(gguf_probe.confidence == 1.0f);
    assert(gguf_probe.paths.size() == 1);
    const auto * gguf_evidence = find_signal(gguf_probe, COMMON_TREE_DRAFT_MODEL_PROBE_GGUF_HEADER);
    assert(gguf_evidence != nullptr);
    assert(gguf_evidence->bytes_read == 24);
    assert(gguf_evidence->file_size == 24);

    const std::string st_header = R"({"weight":{"dtype":"F32","shape":[1024],"data_offsets":[0,4096]}})";
    const fs::path st0 = root / "model-00001-of-00002.safetensors";
    const fs::path st1 = root / "model-00002-of-00002.safetensors";
    write_safetensors_probe_fixture(st0, st_header);
    write_safetensors_probe_fixture(st1, st_header);

    const auto shards_probe = common_tree_draft_model_source_probe(std::vector<std::string>{ st1.string(), st0.string() });
    assert(shards_probe.format == COMMON_TREE_DRAFT_MODEL_FORMAT_SAFETENSORS);
    assert(shards_probe.loader == COMMON_TREE_DRAFT_MODEL_LOADER_SAFETENSORS);
    assert(shards_probe.confidence == 1.0f);
    assert(shards_probe.paths.size() == 2);
    assert(shards_probe.paths[0] < shards_probe.paths[1]);
    for (const auto & item : shards_probe.evidence) {
        if (item.signal == COMMON_TREE_DRAFT_MODEL_PROBE_SAFETENSORS_HEADER) {
            assert(item.bytes_read == 8 + st_header.size());
            assert(item.bytes_read == item.file_size);
        }
    }

    const fs::path index = root / "model.safetensors.index.json";
    {
        std::ofstream file(index, std::ios::binary);
        file << R"({"weight_map":{"a":"model-00002-of-00002.safetensors","b":"model-00001-of-00002.safetensors"}})";
    }
    const auto index_probe = common_tree_draft_model_source_probe(index.string());
    assert(index_probe.format == COMMON_TREE_DRAFT_MODEL_FORMAT_SAFETENSORS);
    assert(index_probe.confidence == 1.0f);
    assert(index_probe.paths == shards_probe.paths);
    assert(find_signal(index_probe, COMMON_TREE_DRAFT_MODEL_PROBE_SAFETENSORS_INDEX) != nullptr);

    const fs::path invalid = root / "invalid.bin";
    write_bytes(invalid, { 'n', 'o', 'p', 'e' });
    const auto invalid_probe = common_tree_draft_model_source_probe(invalid.string());
    assert(invalid_probe.format == COMMON_TREE_DRAFT_MODEL_FORMAT_UNKNOWN);
    assert(invalid_probe.loader == COMMON_TREE_DRAFT_MODEL_LOADER_NONE);
    assert(invalid_probe.confidence == 0.0f);

    const auto mixed_probe = common_tree_draft_model_source_probe(std::vector<std::string>{ gguf.string(), st0.string() });
    assert(mixed_probe.format == COMMON_TREE_DRAFT_MODEL_FORMAT_UNKNOWN);
    assert(mixed_probe.loader == COMMON_TREE_DRAFT_MODEL_LOADER_NONE);
    assert(mixed_probe.confidence == 0.0f);
    assert(find_signal(mixed_probe, COMMON_TREE_DRAFT_MODEL_PROBE_AMBIGUOUS) != nullptr);

    fs::remove_all(root, ec);
    return 0;
}
