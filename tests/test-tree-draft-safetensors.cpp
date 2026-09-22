#include "tree-draft-safetensors.h"

#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static void append_u64_le(std::vector<unsigned char> & bytes, uint64_t value) {
    for (int i = 0; i < 8; ++i) {
        bytes.push_back(static_cast<unsigned char>((value >> (8 * i)) & 0xff));
    }
}

static void write_safetensors(const fs::path & path, const std::string & header, size_t payload_size) {
    std::vector<unsigned char> bytes;
    append_u64_le(bytes, header.size());
    bytes.insert(bytes.end(), header.begin(), header.end());
    bytes.resize(bytes.size() + payload_size);
    std::ofstream file(path, std::ios::binary);
    assert(file);
    file.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    assert(file);
}

static common_tree_draft_model_source source_for(const fs::path & path) {
    common_tree_draft_model_source source;
    source.format = COMMON_TREE_DRAFT_MODEL_FORMAT_SAFETENSORS;
    source.loader = COMMON_TREE_DRAFT_MODEL_LOADER_SAFETENSORS;
    source.confidence = 1.0f;
    source.paths.push_back(path.generic_string());
    return source;
}

static common_tree_draft_safetensors_result parse_fixture(
        const fs::path & path,
        const std::string & header,
        size_t payload_size) {
    write_safetensors(path, header, payload_size);
    return common_tree_draft_safetensors_parse(source_for(path));
}

static void test_valid_header_and_metadata(const fs::path & path) {
    const auto result = parse_fixture(path,
            R"({"__metadata__":{"architecture":"gemma4"},"a":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]},"b":{"dtype":"I16","shape":[4],"data_offsets":[24,32]}})",
            32);
    assert(result);
    assert(result.table.payload_size == 32);
    assert(result.table.metadata.at("architecture") == "gemma4");
    assert(result.table.tensors.size() == 2);
    assert(result.table.tensors[0].name == "a");
    assert(result.table.tensors[0].shape == std::vector<uint64_t>({ 2, 3 }));
    assert(result.table.tensors[1].offset_begin == 24);
}

static void expect_error(
        const fs::path & path,
        const std::string & header,
        size_t payload_size,
        common_tree_draft_safetensors_error expected) {
    const auto result = parse_fixture(path, header, payload_size);
    assert(!result);
    assert(result.error == expected);
}

static void test_invalid_descriptors(const fs::path & path) {
    expect_error(path, R"({"a":{"dtype":"F32","shape":[2],"data_offsets":[0,7]}})", 8, COMMON_TREE_DRAFT_SAFETENSORS_SIZE_MISMATCH);
    expect_error(path, R"({"a":{"dtype":"Q4_0","shape":[2],"data_offsets":[0,2]}})", 2, COMMON_TREE_DRAFT_SAFETENSORS_UNSUPPORTED_DTYPE);
    expect_error(path, R"({"a":{"dtype":"F32","shape":[-1],"data_offsets":[0,4]}})", 4, COMMON_TREE_DRAFT_SAFETENSORS_INVALID_SHAPE);
    expect_error(path, R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[4,0]}})", 4, COMMON_TREE_DRAFT_SAFETENSORS_INVALID_OFFSETS);
    expect_error(path, R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"b":{"dtype":"F32","shape":[1],"data_offsets":[2,6]}})", 6, COMMON_TREE_DRAFT_SAFETENSORS_OVERLAPPING_TENSORS);
    expect_error(path, R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"a":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}})", 8, COMMON_TREE_DRAFT_SAFETENSORS_DUPLICATE_TENSOR);
}

static void test_header_errors(const fs::path & path) {
    expect_error(path, "[1,2,3]", 0, COMMON_TREE_DRAFT_SAFETENSORS_INVALID_HEADER);
    expect_error(path, "{", 0, COMMON_TREE_DRAFT_SAFETENSORS_INVALID_JSON);

    std::ofstream file(path, std::ios::binary);
    assert(file);
    const unsigned char short_length[] = { 1, 2, 3 };
    file.write(reinterpret_cast<const char *>(short_length), sizeof(short_length));
    file.close();
    const auto truncated = common_tree_draft_safetensors_parse(source_for(path));
    assert(!truncated);
    assert(truncated.error == COMMON_TREE_DRAFT_SAFETENSORS_TRUNCATED);
}

int main() {
    const fs::path root = fs::temp_directory_path() / "llama-tree-draft-safetensors-test";
    std::error_code ec;
    fs::remove_all(root, ec);
    assert(fs::create_directories(root));
    const fs::path path = root / "model.safetensors";

    test_valid_header_and_metadata(path);
    test_invalid_descriptors(path);
    test_header_errors(path);

    fs::remove_all(root, ec);
    return 0;
}
