#include "tree-draft-safetensors-index.h"

#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

static void append_u64_le(std::vector<unsigned char> & bytes, uint64_t value) {
    for (int i = 0; i < 8; ++i) {
        bytes.push_back(static_cast<unsigned char>((value >> (8 * i)) & 0xff));
    }
}

static void write_safetensors(
        const fs::path & path,
        const std::vector<std::pair<std::string, std::pair<uint64_t, uint64_t>>> & tensors) {
    std::string header = "{";
    bool first = true;
    uint64_t payload_size = 0;
    for (const auto & item : tensors) {
        if (!first) header += ",";
        first = false;
        header += "\"" + item.first + "\":{\"dtype\":\"F32\",\"shape\":[" +
            std::to_string((item.second.second - item.second.first) / 4) +
            "],\"data_offsets\":[" + std::to_string(item.second.first) + "," +
            std::to_string(item.second.second) + "]}";
        if (item.second.second > payload_size) payload_size = item.second.second;
    }
    header += "}";

    std::vector<unsigned char> bytes;
    append_u64_le(bytes, header.size());
    bytes.insert(bytes.end(), header.begin(), header.end());
    bytes.resize(bytes.size() + static_cast<size_t>(payload_size), 0);

    std::ofstream file(path, std::ios::binary);
    assert(file);
    file.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    assert(file);
}

static void write_text(const fs::path & path, const std::string & text) {
    std::ofstream file(path, std::ios::binary);
    assert(file);
    file << text;
    assert(file);
}

static common_tree_draft_safetensors_index_result resolve(const fs::path & index) {
    const auto source = common_tree_draft_model_source_probe(index.string());
    return common_tree_draft_safetensors_index_resolve(source);
}

int main() {
    const fs::path root = fs::temp_directory_path() / "llama-tree-draft-safetensors-index-test";
    std::error_code ec;
    fs::remove_all(root, ec);
    assert(fs::create_directories(root));

    const fs::path shard0 = root / "model-00001-of-00002.safetensors";
    const fs::path shard1 = root / "model-00002-of-00002.safetensors";
    const fs::path index = root / "model.safetensors.index.json";

    write_safetensors(shard0, { { "z.weight", { 0, 16 } }, { "a.weight", { 16, 32 } } });
    write_safetensors(shard1, { { "m.weight", { 0, 16 } } });
    write_text(index,
        R"({"metadata":{"total_size":48},"weight_map":{"z.weight":"model-00001-of-00002.safetensors","m.weight":"model-00002-of-00002.safetensors","a.weight":"model-00001-of-00002.safetensors"}})");

    const auto valid = resolve(index);
    assert(valid);
    assert(valid.directory.tensors.size() == 3);
    assert(valid.directory.tensors[0].descriptor.name == "a.weight");
    assert(valid.directory.tensors[1].descriptor.name == "m.weight");
    assert(valid.directory.tensors[2].descriptor.name == "z.weight");
    assert(valid.directory.tensors[0].source_path.find("00001") != std::string::npos);
    assert(valid.directory.tensors[1].source_path.find("00002") != std::string::npos);
    assert(valid.directory.shard_paths.size() == 2);

    write_text(index,
        R"({"weight_map":{"a.weight":"missing.safetensors"}})");
    assert(resolve(index).error == COMMON_TREE_DRAFT_SAFETENSORS_INDEX_MISSING_SHARD);

    write_safetensors(shard0, { { "z.weight", { 0, 16 } } });
    write_text(index,
        R"({"weight_map":{"missing.weight":"model-00001-of-00002.safetensors","z.weight":"model-00001-of-00002.safetensors"}})");
    assert(resolve(index).error == COMMON_TREE_DRAFT_SAFETENSORS_INDEX_TENSOR_MISSING);

    write_safetensors(shard0, { { "z.weight", { 0, 16 } }, { "a.weight", { 16, 32 } } });
    write_text(index,
        R"({"weight_map":{"missing.weight":"model-00001-of-00002.safetensors","z.weight":"model-00001-of-00002.safetensors"}})");
    assert(resolve(index).error == COMMON_TREE_DRAFT_SAFETENSORS_INDEX_TENSOR_CONFLICT);

    write_safetensors(shard0, { { "a.weight", { 0, 16 } }, { "extra.weight", { 16, 32 } } });
    write_text(index,
        R"({"weight_map":{"a.weight":"model-00001-of-00002.safetensors"}})");
    assert(resolve(index).error == COMMON_TREE_DRAFT_SAFETENSORS_INDEX_TENSOR_CONFLICT);

    write_safetensors(shard0, { { "a.weight", { 0, 16 } } });
    write_safetensors(shard1, { { "m.weight", { 0, 16 } } });
    write_text(index,
        R"({"weight_map":{"m.weight":"model-00001-of-00002.safetensors"}})");
    assert(resolve(index).error == COMMON_TREE_DRAFT_SAFETENSORS_INDEX_TENSOR_CONFLICT);

    write_text(index,
        R"({"weight_map":{"a.weight":"model-00001-of-00002.safetensors","a.weight":"model-00001-of-00002.safetensors"}})");
    assert(resolve(index).error == COMMON_TREE_DRAFT_SAFETENSORS_INDEX_DUPLICATE_TENSOR);

    write_text(index,
        R"({"metadata":{"a.weight":"allowed here"},"weight_map":{"a.weight":"model-00001-of-00002.safetensors"}})");
    assert(resolve(index));

    fs::remove_all(root, ec);
    return 0;
}
