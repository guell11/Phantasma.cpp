#include "tree-draft-gguf-header.h"

#include "gguf.h"

#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
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

static std::vector<unsigned char> make_header(
        uint32_t version,
        uint64_t n_tensors = 0,
        uint64_t n_kv = 0,
        bool valid_magic = true) {
    std::vector<unsigned char> bytes;
    if (valid_magic) {
        bytes.insert(bytes.end(), GGUF_MAGIC, GGUF_MAGIC + 4);
    } else {
        const unsigned char bad_magic[] = { 'F', 'U', 'G', 'G' };
        bytes.insert(bytes.end(), bad_magic, bad_magic + 4);
    }
    append_u32_le(bytes, version);
    append_u64_le(bytes, n_tensors);
    append_u64_le(bytes, n_kv);
    return bytes;
}

static void write_bytes(const fs::path & path, const std::vector<unsigned char> & bytes) {
    std::ofstream file(path, std::ios::binary);
    assert(file);
    file.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    assert(file);
}

static common_tree_draft_model_source gguf_source(const fs::path & path) {
    common_tree_draft_model_source source;
    source.format = COMMON_TREE_DRAFT_MODEL_FORMAT_GGUF;
    source.loader = COMMON_TREE_DRAFT_MODEL_LOADER_GGUF;
    source.confidence = 1.0f;
    source.paths.push_back(path.generic_string());
    return source;
}

static void expect_error(
        const fs::path & path,
        const std::vector<unsigned char> & bytes,
        common_tree_draft_gguf_header_error expected) {
    write_bytes(path, bytes);
    const auto result = common_tree_draft_gguf_header_parse(gguf_source(path));
    assert(!result);
    assert(result.error == expected);
}

int main() {
    const fs::path root = fs::temp_directory_path() / "llama-tree-draft-gguf-header-test";
    std::error_code ec;
    fs::remove_all(root, ec);
    assert(fs::create_directories(root));

    for (uint32_t version : { 2u, static_cast<uint32_t>(GGUF_VERSION) }) {
        const fs::path path = root / ("valid-v" + std::to_string(version) + ".gguf");
        write_bytes(path, make_header(version, 7, 11));
        const auto source = common_tree_draft_model_source_probe(path.string());
        assert(source.format == COMMON_TREE_DRAFT_MODEL_FORMAT_GGUF);
        assert(source.loader == COMMON_TREE_DRAFT_MODEL_LOADER_GGUF);
        const auto result = common_tree_draft_gguf_header_parse(source);
        assert(result);
        assert(result.header.version == version);
        assert(result.header.n_tensors == 7);
        assert(result.header.n_kv == 11);
        assert(result.header.header_end == 24);
    }

    const fs::path malformed = root / "malformed.gguf";
    expect_error(malformed, std::vector<unsigned char>{ 'G', 'G', 'U' }, COMMON_TREE_DRAFT_GGUF_HEADER_TRUNCATED);
    expect_error(malformed, make_header(GGUF_VERSION, 0, 0, false), COMMON_TREE_DRAFT_GGUF_HEADER_BAD_MAGIC);
    expect_error(malformed, make_header(0), COMMON_TREE_DRAFT_GGUF_HEADER_UNSUPPORTED_VERSION);
    expect_error(malformed, make_header(1), COMMON_TREE_DRAFT_GGUF_HEADER_UNSUPPORTED_VERSION);
    expect_error(malformed, make_header(GGUF_VERSION + 1), COMMON_TREE_DRAFT_GGUF_HEADER_UNSUPPORTED_VERSION);
    expect_error(malformed, make_header(GGUF_VERSION, UINT64_MAX, 0), COMMON_TREE_DRAFT_GGUF_HEADER_IMPOSSIBLE_COUNT);
    expect_error(malformed, make_header(GGUF_VERSION, 0, UINT64_MAX), COMMON_TREE_DRAFT_GGUF_HEADER_IMPOSSIBLE_COUNT);

    auto truncated = make_header(GGUF_VERSION, 1, 1);
    truncated.pop_back();
    expect_error(malformed, truncated, COMMON_TREE_DRAFT_GGUF_HEADER_TRUNCATED);

    common_tree_draft_model_source wrong_source;
    wrong_source.format = COMMON_TREE_DRAFT_MODEL_FORMAT_SAFETENSORS;
    wrong_source.loader = COMMON_TREE_DRAFT_MODEL_LOADER_SAFETENSORS;
    wrong_source.paths.push_back(malformed.generic_string());
    const auto wrong = common_tree_draft_gguf_header_parse(wrong_source);
    assert(!wrong);
    assert(wrong.error == COMMON_TREE_DRAFT_GGUF_HEADER_INVALID_SOURCE);

    fs::remove_all(root, ec);
    return 0;
}
