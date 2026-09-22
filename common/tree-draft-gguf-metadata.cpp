#include "tree-draft-gguf-metadata.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <new>
#include <unordered_set>

namespace fs = std::filesystem;

namespace {

struct bounded_reader {
    std::ifstream file;
    uint64_t size = 0;
    uint64_t offset = 0;

    bounded_reader(const fs::path & path, uint64_t file_size, uint64_t start)
        : file(path, std::ios::binary), size(file_size), offset(start) {
        if (file) {
            file.seekg(static_cast<std::streamoff>(start), std::ios::beg);
        }
    }

    bool read(void * dst, uint64_t n) {
        if (!file || offset > size || n > size - offset ||
                n > static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max())) {
            return false;
        }
        file.read(static_cast<char *>(dst), static_cast<std::streamsize>(n));
        if (static_cast<uint64_t>(file.gcount()) != n) {
            return false;
        }
        offset += n;
        return true;
    }

    uint64_t remaining() const {
        return offset <= size ? size - offset : 0;
    }
};

uint32_t decode_u32_le(const unsigned char * bytes) {
    uint32_t value = 0;
    for (size_t i = 0; i < sizeof(value); ++i) {
        value |= static_cast<uint32_t>(bytes[i]) << (8 * i);
    }
    return value;
}

uint64_t decode_u64_le(const unsigned char * bytes) {
    uint64_t value = 0;
    for (size_t i = 0; i < sizeof(value); ++i) {
        value |= static_cast<uint64_t>(bytes[i]) << (8 * i);
    }
    return value;
}

bool read_u32(bounded_reader & reader, uint32_t & value) {
    unsigned char bytes[4];
    if (!reader.read(bytes, sizeof(bytes))) {
        return false;
    }
    value = decode_u32_le(bytes);
    return true;
}

bool read_u64(bounded_reader & reader, uint64_t & value) {
    unsigned char bytes[8];
    if (!reader.read(bytes, sizeof(bytes))) {
        return false;
    }
    value = decode_u64_le(bytes);
    return true;
}

common_tree_draft_gguf_metadata_result fail(
        common_tree_draft_gguf_metadata_error error,
        uint64_t offset,
        const char * message) {
    common_tree_draft_gguf_metadata_result result;
    result.error = error;
    result.error_offset = offset;
    result.message = message;
    return result;
}

bool type_is_scalar(gguf_type type) {
    return type >= GGUF_TYPE_UINT8 && type < GGUF_TYPE_COUNT && type != GGUF_TYPE_ARRAY;
}

bool type_is_array_element(gguf_type type) {
    return type_is_scalar(type);
}

size_t metadata_type_size(gguf_type type) {
    switch (type) {
        case GGUF_TYPE_UINT8:   return sizeof(uint8_t);
        case GGUF_TYPE_INT8:    return sizeof(int8_t);
        case GGUF_TYPE_UINT16:  return sizeof(uint16_t);
        case GGUF_TYPE_INT16:   return sizeof(int16_t);
        case GGUF_TYPE_UINT32:  return sizeof(uint32_t);
        case GGUF_TYPE_INT32:   return sizeof(int32_t);
        case GGUF_TYPE_FLOAT32: return sizeof(float);
        case GGUF_TYPE_BOOL:    return sizeof(int8_t);
        case GGUF_TYPE_UINT64:  return sizeof(uint64_t);
        case GGUF_TYPE_INT64:   return sizeof(int64_t);
        case GGUF_TYPE_FLOAT64: return sizeof(double);
        case GGUF_TYPE_STRING:
        case GGUF_TYPE_ARRAY:
        case GGUF_TYPE_COUNT:
            return 0;
    }
    return 0;
}

bool read_string(
        bounded_reader & reader,
        std::string & value,
        common_tree_draft_gguf_metadata_result & error) {
    const uint64_t length_offset = reader.offset;
    uint64_t length = 0;
    if (!read_u64(reader, length)) {
        error = fail(COMMON_TREE_DRAFT_GGUF_METADATA_TRUNCATED, length_offset, "truncated GGUF string length");
        return false;
    }
    if (length > reader.remaining() || length > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        error = fail(COMMON_TREE_DRAFT_GGUF_METADATA_LIMIT_EXCEEDED, length_offset, "GGUF string exceeds file bounds");
        return false;
    }
    try {
        value.resize(static_cast<size_t>(length));
    } catch (const std::bad_alloc &) {
        error = fail(COMMON_TREE_DRAFT_GGUF_METADATA_LIMIT_EXCEEDED, length_offset, "GGUF string allocation failed");
        return false;
    } catch (const std::length_error &) {
        error = fail(COMMON_TREE_DRAFT_GGUF_METADATA_LIMIT_EXCEEDED, length_offset, "GGUF string length is not representable");
        return false;
    }
    if (length > 0 && !reader.read(value.data(), length)) {
        error = fail(COMMON_TREE_DRAFT_GGUF_METADATA_TRUNCATED, reader.offset, "truncated GGUF string bytes");
        return false;
    }
    return true;
}

bool read_raw_values(
        bounded_reader & reader,
        gguf_type type,
        uint64_t count,
        std::vector<unsigned char> & raw,
        common_tree_draft_gguf_metadata_result & error) {
    const size_t item_size = metadata_type_size(type);
    if (item_size == 0) {
        error = fail(COMMON_TREE_DRAFT_GGUF_METADATA_UNSUPPORTED_TYPE, reader.offset, "unsupported GGUF value type");
        return false;
    }
    if (count > std::numeric_limits<uint64_t>::max() / item_size) {
        error = fail(COMMON_TREE_DRAFT_GGUF_METADATA_LIMIT_EXCEEDED, reader.offset, "GGUF array byte count overflows");
        return false;
    }
    const uint64_t byte_count = count * item_size;
    if (byte_count > reader.remaining() || byte_count > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        error = fail(COMMON_TREE_DRAFT_GGUF_METADATA_LIMIT_EXCEEDED, reader.offset, "GGUF value exceeds file bounds");
        return false;
    }
    try {
        raw.resize(static_cast<size_t>(byte_count));
    } catch (const std::bad_alloc &) {
        error = fail(COMMON_TREE_DRAFT_GGUF_METADATA_LIMIT_EXCEEDED, reader.offset, "GGUF value allocation failed");
        return false;
    } catch (const std::length_error &) {
        error = fail(COMMON_TREE_DRAFT_GGUF_METADATA_LIMIT_EXCEEDED, reader.offset, "GGUF value size is not representable");
        return false;
    }
    if (byte_count > 0 && !reader.read(raw.data(), byte_count)) {
        error = fail(COMMON_TREE_DRAFT_GGUF_METADATA_TRUNCATED, reader.offset, "truncated GGUF value bytes");
        return false;
    }
    return true;
}

common_tree_draft_gguf_metadata_lookup_result lookup_type(
        const common_tree_draft_gguf_metadata_table & table,
        const std::string & key,
        bool expect_array,
        gguf_type expected_type) {
    if ((expect_array && !type_is_array_element(expected_type)) || (!expect_array && !type_is_scalar(expected_type))) {
        return { COMMON_TREE_DRAFT_GGUF_METADATA_LOOKUP_UNSUPPORTED, nullptr };
    }
    const auto found = common_tree_draft_gguf_metadata_find(table, key);
    if (!found) {
        return found;
    }
    const auto & value = found.entry->value;
    if (value.is_array != expect_array ||
            (expect_array ? value.array_type : value.type) != expected_type) {
        return { COMMON_TREE_DRAFT_GGUF_METADATA_LOOKUP_MALFORMED, found.entry };
    }
    return found;
}

}

common_tree_draft_gguf_metadata_result common_tree_draft_gguf_metadata_parse(
        const common_tree_draft_model_source & source,
        size_t path_index) {
    const auto header = common_tree_draft_gguf_header_parse(source, path_index);
    if (!header) {
        if (header.error == COMMON_TREE_DRAFT_GGUF_HEADER_INVALID_SOURCE) {
            return fail(COMMON_TREE_DRAFT_GGUF_METADATA_INVALID_SOURCE, header.error_offset, header.message.c_str());
        }
        auto result = fail(COMMON_TREE_DRAFT_GGUF_METADATA_HEADER_ERROR, header.error_offset, header.message.c_str());
        result.header_error = header.error;
        return result;
    }

    if (path_index >= source.paths.size()) {
        return fail(COMMON_TREE_DRAFT_GGUF_METADATA_INVALID_SOURCE, 0, "GGUF source path is missing");
    }

    const fs::path path(source.paths[path_index]);
    std::error_code ec;
    const uintmax_t native_size = fs::file_size(path, ec);
    if (ec || native_size > std::numeric_limits<uint64_t>::max()) {
        return fail(COMMON_TREE_DRAFT_GGUF_METADATA_OPEN_FAILED, 0, "failed to obtain GGUF file size");
    }
    const uint64_t file_size = static_cast<uint64_t>(native_size);
    bounded_reader reader(path, file_size, header.header.header_end);
    if (!reader.file) {
        return fail(COMMON_TREE_DRAFT_GGUF_METADATA_OPEN_FAILED, 0, "failed to open GGUF metadata source");
    }

    static constexpr uint64_t MIN_METADATA_ENTRY_SIZE = sizeof(uint64_t) + 1 + sizeof(uint32_t) + sizeof(uint8_t);
    if (header.header.n_kv > reader.remaining() / MIN_METADATA_ENTRY_SIZE) {
        return fail(COMMON_TREE_DRAFT_GGUF_METADATA_LIMIT_EXCEEDED, reader.offset, "GGUF metadata count exceeds file bounds");
    }

    common_tree_draft_gguf_metadata_result result;
    result.table.entries.reserve(static_cast<size_t>(header.header.n_kv));
    std::unordered_set<std::string> keys;
    keys.reserve(static_cast<size_t>(header.header.n_kv));

    for (uint64_t i = 0; i < header.header.n_kv; ++i) {
        common_tree_draft_gguf_metadata_entry entry;
        if (!read_string(reader, entry.key, result)) {
            return result;
        }
        if (entry.key.empty()) {
            return fail(COMMON_TREE_DRAFT_GGUF_METADATA_EMPTY_KEY, reader.offset, "GGUF metadata key is empty");
        }
        if (!keys.insert(entry.key).second) {
            return fail(COMMON_TREE_DRAFT_GGUF_METADATA_DUPLICATE_KEY, reader.offset, "duplicate GGUF metadata key");
        }

        const uint64_t type_offset = reader.offset;
        uint32_t type_raw = 0;
        if (!read_u32(reader, type_raw)) {
            return fail(COMMON_TREE_DRAFT_GGUF_METADATA_TRUNCATED, type_offset, "truncated GGUF metadata type");
        }
        if (type_raw >= static_cast<uint32_t>(GGUF_TYPE_COUNT)) {
            return fail(COMMON_TREE_DRAFT_GGUF_METADATA_INVALID_TYPE, type_offset, "invalid GGUF metadata type");
        }

        entry.value.type = static_cast<gguf_type>(type_raw);
        if (entry.value.type == GGUF_TYPE_ARRAY) {
            entry.value.is_array = true;
            uint32_t element_type_raw = 0;
            const uint64_t element_type_offset = reader.offset;
            if (!read_u32(reader, element_type_raw)) {
                return fail(COMMON_TREE_DRAFT_GGUF_METADATA_TRUNCATED, element_type_offset, "truncated GGUF array element type");
            }
            if (element_type_raw >= static_cast<uint32_t>(GGUF_TYPE_COUNT)) {
                return fail(COMMON_TREE_DRAFT_GGUF_METADATA_INVALID_TYPE, element_type_offset, "invalid GGUF array element type");
            }
            entry.value.array_type = static_cast<gguf_type>(element_type_raw);
            if (!type_is_array_element(entry.value.array_type)) {
                return fail(COMMON_TREE_DRAFT_GGUF_METADATA_UNSUPPORTED_TYPE, element_type_offset, "nested GGUF arrays are unsupported");
            }

            const uint64_t count_offset = reader.offset;
            uint64_t count = 0;
            if (!read_u64(reader, count)) {
                return fail(COMMON_TREE_DRAFT_GGUF_METADATA_TRUNCATED, count_offset, "truncated GGUF array length");
            }

            if (entry.value.array_type == GGUF_TYPE_STRING) {
                if (count > reader.remaining() / sizeof(uint64_t) ||
                        count > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
                    return fail(COMMON_TREE_DRAFT_GGUF_METADATA_LIMIT_EXCEEDED, count_offset, "GGUF string array length exceeds file bounds");
                }
                try {
                    entry.value.strings.reserve(static_cast<size_t>(count));
                } catch (const std::bad_alloc &) {
                    return fail(COMMON_TREE_DRAFT_GGUF_METADATA_LIMIT_EXCEEDED, count_offset, "GGUF string array allocation failed");
                } catch (const std::length_error &) {
                    return fail(COMMON_TREE_DRAFT_GGUF_METADATA_LIMIT_EXCEEDED, count_offset, "GGUF string array length is not representable");
                }
                for (uint64_t j = 0; j < count; ++j) {
                    std::string value;
                    if (!read_string(reader, value, result)) {
                        return result;
                    }
                    entry.value.strings.push_back(std::move(value));
                }
            } else if (!read_raw_values(reader, entry.value.array_type, count, entry.value.raw, result)) {
                return result;
            }
        } else {
            entry.value.array_type = GGUF_TYPE_COUNT;
            if (entry.value.type == GGUF_TYPE_STRING) {
                std::string value;
                if (!read_string(reader, value, result)) {
                    return result;
                }
                entry.value.strings.push_back(std::move(value));
            } else if (!type_is_scalar(entry.value.type)) {
                return fail(COMMON_TREE_DRAFT_GGUF_METADATA_UNSUPPORTED_TYPE, type_offset, "unsupported GGUF metadata type");
            } else if (!read_raw_values(reader, entry.value.type, 1, entry.value.raw, result)) {
                return result;
            }
        }

        result.table.entries.push_back(std::move(entry));
    }

    result.table.metadata_end = reader.offset;
    return result;
}

common_tree_draft_gguf_metadata_lookup_result common_tree_draft_gguf_metadata_find(
        const common_tree_draft_gguf_metadata_table & table,
        const std::string & key) {
    for (const auto & entry : table.entries) {
        if (entry.key == key) {
            return { COMMON_TREE_DRAFT_GGUF_METADATA_LOOKUP_OK, &entry };
        }
    }
    return { COMMON_TREE_DRAFT_GGUF_METADATA_LOOKUP_MISSING, nullptr };
}

common_tree_draft_gguf_metadata_lookup_result common_tree_draft_gguf_metadata_get_scalar(
        const common_tree_draft_gguf_metadata_table & table,
        const std::string & key,
        gguf_type expected_type) {
    return lookup_type(table, key, false, expected_type);
}

common_tree_draft_gguf_metadata_lookup_result common_tree_draft_gguf_metadata_get_array(
        const common_tree_draft_gguf_metadata_table & table,
        const std::string & key,
        gguf_type expected_element_type) {
    return lookup_type(table, key, true, expected_element_type);
}

const char * common_tree_draft_gguf_metadata_error_name(common_tree_draft_gguf_metadata_error error) {
    switch (error) {
        case COMMON_TREE_DRAFT_GGUF_METADATA_OK:               return "ok";
        case COMMON_TREE_DRAFT_GGUF_METADATA_INVALID_SOURCE:   return "invalid_source";
        case COMMON_TREE_DRAFT_GGUF_METADATA_HEADER_ERROR:      return "header_error";
        case COMMON_TREE_DRAFT_GGUF_METADATA_OPEN_FAILED:       return "open_failed";
        case COMMON_TREE_DRAFT_GGUF_METADATA_TRUNCATED:         return "truncated";
        case COMMON_TREE_DRAFT_GGUF_METADATA_EMPTY_KEY:         return "empty_key";
        case COMMON_TREE_DRAFT_GGUF_METADATA_DUPLICATE_KEY:     return "duplicate_key";
        case COMMON_TREE_DRAFT_GGUF_METADATA_INVALID_TYPE:      return "invalid_type";
        case COMMON_TREE_DRAFT_GGUF_METADATA_UNSUPPORTED_TYPE:  return "unsupported_type";
        case COMMON_TREE_DRAFT_GGUF_METADATA_LIMIT_EXCEEDED:    return "limit_exceeded";
    }
    return "unknown";
}

const char * common_tree_draft_gguf_metadata_lookup_status_name(common_tree_draft_gguf_metadata_lookup_status status) {
    switch (status) {
        case COMMON_TREE_DRAFT_GGUF_METADATA_LOOKUP_OK:          return "ok";
        case COMMON_TREE_DRAFT_GGUF_METADATA_LOOKUP_MISSING:     return "missing";
        case COMMON_TREE_DRAFT_GGUF_METADATA_LOOKUP_MALFORMED:   return "malformed";
        case COMMON_TREE_DRAFT_GGUF_METADATA_LOOKUP_UNSUPPORTED: return "unsupported";
    }
    return "unknown";
}
