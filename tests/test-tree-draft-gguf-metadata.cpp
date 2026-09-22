#include "tree-draft-gguf-metadata.h"

#include "gguf.h"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static void append_u32_le(std::vector<unsigned char> & bytes, uint32_t value) {
    for (int i = 0; i < 4; ++i) {
        bytes.push_back(static_cast<unsigned char>((value >> (8 * i)) & 0xff));
    }
}

static void append_u64_le(std::vector<unsigned char> & bytes, uint64_t value) {
    for (int i = 0; i < 8; ++i) {
        bytes.push_back(static_cast<unsigned char>((value >> (8 * i)) & 0xff));
    }
}

static void append_string(std::vector<unsigned char> & bytes, const std::string & value) {
    append_u64_le(bytes, value.size());
    bytes.insert(bytes.end(), value.begin(), value.end());
}

static std::vector<unsigned char> make_header(uint64_t n_kv) {
    std::vector<unsigned char> bytes(GGUF_MAGIC, GGUF_MAGIC + 4);
    append_u32_le(bytes, GGUF_VERSION);
    append_u64_le(bytes, 0);
    append_u64_le(bytes, n_kv);
    return bytes;
}

static void append_scalar_u32(
        std::vector<unsigned char> & bytes,
        const std::string & key,
        uint32_t value) {
    append_string(bytes, key);
    append_u32_le(bytes, GGUF_TYPE_UINT32);
    append_u32_le(bytes, value);
}

static void append_scalar_u64(
        std::vector<unsigned char> & bytes,
        const std::string & key,
        uint64_t value) {
    append_string(bytes, key);
    append_u32_le(bytes, GGUF_TYPE_UINT64);
    append_u64_le(bytes, value);
}

static void append_scalar_string(
        std::vector<unsigned char> & bytes,
        const std::string & key,
        const std::string & value) {
    append_string(bytes, key);
    append_u32_le(bytes, GGUF_TYPE_STRING);
    append_string(bytes, value);
}

static void append_i32_array(
        std::vector<unsigned char> & bytes,
        const std::string & key,
        const std::vector<int32_t> & values) {
    append_string(bytes, key);
    append_u32_le(bytes, GGUF_TYPE_ARRAY);
    append_u32_le(bytes, GGUF_TYPE_INT32);
    append_u64_le(bytes, values.size());
    for (int32_t value : values) {
        append_u32_le(bytes, static_cast<uint32_t>(value));
    }
}

static void append_string_array(
        std::vector<unsigned char> & bytes,
        const std::string & key,
        const std::vector<std::string> & values) {
    append_string(bytes, key);
    append_u32_le(bytes, GGUF_TYPE_ARRAY);
    append_u32_le(bytes, GGUF_TYPE_STRING);
    append_u64_le(bytes, values.size());
    for (const auto & value : values) {
        append_string(bytes, value);
    }
}

static void write_bytes(const fs::path & path, const std::vector<unsigned char> & bytes) {
    std::ofstream file(path, std::ios::binary);
    assert(file);
    file.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    assert(file);
}

static common_tree_draft_gguf_metadata_result parse_fixture(
        const fs::path & path,
        const std::vector<unsigned char> & bytes) {
    write_bytes(path, bytes);
    const auto source = common_tree_draft_model_source_probe(path.string());
    assert(source.format == COMMON_TREE_DRAFT_MODEL_FORMAT_GGUF);
    assert(source.loader == COMMON_TREE_DRAFT_MODEL_LOADER_GGUF);
    return common_tree_draft_gguf_metadata_parse(source);
}

static uint32_t raw_u32(const common_tree_draft_gguf_metadata_entry & entry) {
    assert(entry.value.raw.size() == sizeof(uint32_t));
    const auto & raw = entry.value.raw;
    return static_cast<uint32_t>(raw[0]) |
        (static_cast<uint32_t>(raw[1]) << 8) |
        (static_cast<uint32_t>(raw[2]) << 16) |
        (static_cast<uint32_t>(raw[3]) << 24);
}

int main() {
    const fs::path root = fs::temp_directory_path() / "llama-tree-draft-gguf-metadata-test";
    std::error_code ec;
    fs::remove_all(root, ec);
    assert(fs::create_directories(root));
    const fs::path path = root / "metadata.gguf";

    std::vector<unsigned char> valid = make_header(5);
    append_scalar_u32(valid, "general.alignment", 64);
    append_scalar_string(valid, "general.architecture", "gemma4");
    append_i32_array(valid, "split.indices", { 7, -2, 19 });
    append_string_array(valid, "tokenizer.tags", { "alpha", "beta" });
    append_scalar_u64(valid, "vendor.experimental.unknown_key", UINT64_C(0x1122334455667788));

    const auto parsed = parse_fixture(path, valid);
    assert(parsed);
    assert(parsed.table.entries.size() == 5);
    assert(parsed.table.metadata_end == valid.size());

    const auto alignment = common_tree_draft_gguf_metadata_get_scalar(parsed.table, "general.alignment", GGUF_TYPE_UINT32);
    assert(alignment);
    assert(raw_u32(*alignment.entry) == 64);

    const auto architecture = common_tree_draft_gguf_metadata_get_scalar(parsed.table, "general.architecture", GGUF_TYPE_STRING);
    assert(architecture);
    assert(architecture.entry->value.strings.size() == 1);
    assert(architecture.entry->value.strings[0] == "gemma4");

    const auto indices = common_tree_draft_gguf_metadata_get_array(parsed.table, "split.indices", GGUF_TYPE_INT32);
    assert(indices);
    assert(indices.entry->value.raw.size() == 3 * sizeof(int32_t));

    const auto tags = common_tree_draft_gguf_metadata_get_array(parsed.table, "tokenizer.tags", GGUF_TYPE_STRING);
    assert(tags);
    assert(tags.entry->value.strings == std::vector<std::string>({ "alpha", "beta" }));

    const auto unknown = common_tree_draft_gguf_metadata_find(parsed.table, "vendor.experimental.unknown_key");
    assert(unknown);
    assert(unknown.entry->value.type == GGUF_TYPE_UINT64);
    assert(unknown.entry->value.raw.size() == sizeof(uint64_t));

    const auto missing = common_tree_draft_gguf_metadata_get_scalar(parsed.table, "missing.key", GGUF_TYPE_UINT32);
    assert(missing.status == COMMON_TREE_DRAFT_GGUF_METADATA_LOOKUP_MISSING);
    assert(missing.entry == nullptr);

    const auto malformed = common_tree_draft_gguf_metadata_get_scalar(parsed.table, "general.alignment", GGUF_TYPE_STRING);
    assert(malformed.status == COMMON_TREE_DRAFT_GGUF_METADATA_LOOKUP_MALFORMED);
    assert(malformed.entry != nullptr);

    const auto unsupported = common_tree_draft_gguf_metadata_get_array(parsed.table, "split.indices", GGUF_TYPE_ARRAY);
    assert(unsupported.status == COMMON_TREE_DRAFT_GGUF_METADATA_LOOKUP_UNSUPPORTED);
    assert(unsupported.entry == nullptr);

    std::vector<unsigned char> duplicate = make_header(2);
    append_scalar_u32(duplicate, "dup", 1);
    append_scalar_u32(duplicate, "dup", 2);
    const auto duplicate_result = parse_fixture(path, duplicate);
    assert(!duplicate_result);
    assert(duplicate_result.error == COMMON_TREE_DRAFT_GGUF_METADATA_DUPLICATE_KEY);

    std::vector<unsigned char> oversized_string = make_header(1);
    append_string(oversized_string, "too.long");
    append_u32_le(oversized_string, GGUF_TYPE_STRING);
    append_u64_le(oversized_string, 1024);
    const auto oversized_result = parse_fixture(path, oversized_string);
    assert(!oversized_result);
    assert(oversized_result.error == COMMON_TREE_DRAFT_GGUF_METADATA_LIMIT_EXCEEDED);

    std::vector<unsigned char> invalid_type = make_header(1);
    append_string(invalid_type, "bad.type");
    append_u32_le(invalid_type, GGUF_TYPE_COUNT + 9);
    const auto invalid_type_result = parse_fixture(path, invalid_type);
    assert(!invalid_type_result);
    assert(invalid_type_result.error == COMMON_TREE_DRAFT_GGUF_METADATA_INVALID_TYPE);

    std::vector<unsigned char> nested_array = make_header(1);
    append_string(nested_array, "nested");
    append_u32_le(nested_array, GGUF_TYPE_ARRAY);
    append_u32_le(nested_array, GGUF_TYPE_ARRAY);
    append_u64_le(nested_array, 0);
    const auto nested_result = parse_fixture(path, nested_array);
    assert(!nested_result);
    assert(nested_result.error == COMMON_TREE_DRAFT_GGUF_METADATA_UNSUPPORTED_TYPE);

    const auto impossible_count = parse_fixture(path, make_header(1000));
    assert(!impossible_count);
    assert(impossible_count.error == COMMON_TREE_DRAFT_GGUF_METADATA_LIMIT_EXCEEDED);

    std::vector<unsigned char> truncated = make_header(1);
    append_string(truncated, "cut");
    append_u32_le(truncated, GGUF_TYPE_UINT64);
    append_u32_le(truncated, 7);
    const auto truncated_result = parse_fixture(path, truncated);
    assert(!truncated_result);
    assert(truncated_result.error == COMMON_TREE_DRAFT_GGUF_METADATA_LIMIT_EXCEEDED);

    fs::remove_all(root, ec);
    return 0;
}
