#include "tree-draft-gguf-header.h"

#include "gguf.h"

#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>

namespace fs = std::filesystem;

static constexpr uint64_t GGUF_FIXED_HEADER_SIZE = 4 + sizeof(uint32_t) + 2 * sizeof(uint64_t);

static uint32_t read_u32_le(const unsigned char * bytes) {
    uint32_t value = 0;
    for (size_t i = 0; i < sizeof(value); ++i) {
        value |= static_cast<uint32_t>(bytes[i]) << (8 * i);
    }
    return value;
}

static uint64_t read_u64_le(const unsigned char * bytes) {
    uint64_t value = 0;
    for (size_t i = 0; i < sizeof(value); ++i) {
        value |= static_cast<uint64_t>(bytes[i]) << (8 * i);
    }
    return value;
}

static common_tree_draft_gguf_header_result fail(
        common_tree_draft_gguf_header_error error,
        uint64_t offset,
        const char * message) {
    common_tree_draft_gguf_header_result result;
    result.error = error;
    result.error_offset = offset;
    result.message = message;
    return result;
}

static bool checked_advance(uint64_t offset, uint64_t amount, uint64_t limit, uint64_t & next) {
    if (offset > limit || amount > limit - offset) {
        return false;
    }
    next = offset + amount;
    return true;
}

bool common_tree_draft_gguf_version_supported(uint32_t version) {
    return version >= 2 && version <= GGUF_VERSION;
}

common_tree_draft_gguf_header_result common_tree_draft_gguf_header_parse(
        const common_tree_draft_model_source & source,
        size_t path_index) {
    if (source.format != COMMON_TREE_DRAFT_MODEL_FORMAT_GGUF ||
            source.loader != COMMON_TREE_DRAFT_MODEL_LOADER_GGUF ||
            path_index >= source.paths.size()) {
        return fail(COMMON_TREE_DRAFT_GGUF_HEADER_INVALID_SOURCE, 0, "source is not a selected GGUF input");
    }

    const fs::path path(source.paths[path_index]);
    std::error_code ec;
    const uintmax_t native_size = fs::file_size(path, ec);
    if (ec || native_size > std::numeric_limits<uint64_t>::max()) {
        return fail(COMMON_TREE_DRAFT_GGUF_HEADER_OPEN_FAILED, 0, "failed to obtain GGUF file size");
    }
    const uint64_t file_size = static_cast<uint64_t>(native_size);

    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return fail(COMMON_TREE_DRAFT_GGUF_HEADER_OPEN_FAILED, 0, "failed to open GGUF file");
    }

    std::array<unsigned char, GGUF_FIXED_HEADER_SIZE> bytes{};
    uint64_t header_end = 0;
    if (!checked_advance(0, bytes.size(), file_size, header_end)) {
        return fail(COMMON_TREE_DRAFT_GGUF_HEADER_TRUNCATED, file_size, "GGUF fixed header is truncated");
    }

    file.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (static_cast<size_t>(file.gcount()) != bytes.size()) {
        return fail(COMMON_TREE_DRAFT_GGUF_HEADER_TRUNCATED, static_cast<uint64_t>(file.gcount()), "GGUF fixed header is truncated");
    }

    if (std::memcmp(bytes.data(), GGUF_MAGIC, 4) != 0) {
        return fail(COMMON_TREE_DRAFT_GGUF_HEADER_BAD_MAGIC, 0, "invalid GGUF magic");
    }

    const uint32_t version = read_u32_le(bytes.data() + 4);
    if (!common_tree_draft_gguf_version_supported(version)) {
        return fail(COMMON_TREE_DRAFT_GGUF_HEADER_UNSUPPORTED_VERSION, 4, "unsupported GGUF version");
    }

    const uint64_t n_tensors_raw = read_u64_le(bytes.data() + 8);
    const uint64_t n_kv_raw = read_u64_le(bytes.data() + 16);
    if (n_tensors_raw > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
            n_kv_raw > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
            n_tensors_raw > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
            n_kv_raw > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        return fail(
                COMMON_TREE_DRAFT_GGUF_HEADER_IMPOSSIBLE_COUNT,
                n_tensors_raw > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ? 8 : 16,
                "GGUF header contains an impossible count");
    }

    common_tree_draft_gguf_header_result result;
    result.header.version = version;
    result.header.n_tensors = n_tensors_raw;
    result.header.n_kv = n_kv_raw;
    result.header.header_end = header_end;
    return result;
}

const char * common_tree_draft_gguf_header_error_name(common_tree_draft_gguf_header_error error) {
    switch (error) {
        case COMMON_TREE_DRAFT_GGUF_HEADER_OK:                  return "ok";
        case COMMON_TREE_DRAFT_GGUF_HEADER_INVALID_SOURCE:      return "invalid_source";
        case COMMON_TREE_DRAFT_GGUF_HEADER_OPEN_FAILED:         return "open_failed";
        case COMMON_TREE_DRAFT_GGUF_HEADER_TRUNCATED:           return "truncated";
        case COMMON_TREE_DRAFT_GGUF_HEADER_BAD_MAGIC:           return "bad_magic";
        case COMMON_TREE_DRAFT_GGUF_HEADER_UNSUPPORTED_VERSION: return "unsupported_version";
        case COMMON_TREE_DRAFT_GGUF_HEADER_IMPOSSIBLE_COUNT:    return "impossible_count";
    }
    return "unknown";
}
