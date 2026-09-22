#include "tree-draft-gguf-tensors.h"

#include "gguf.h"

#include <filesystem>
#include <fstream>
#include <limits>
#include <new>
#include <unordered_set>

namespace fs = std::filesystem;

namespace {

struct reader {
    std::ifstream file;
    uint64_t size = 0;
    uint64_t offset = 0;

    reader(const fs::path & path, uint64_t file_size, uint64_t start)
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

    bool skip(uint64_t n) {
        if (offset > size || n > size - offset) {
            return false;
        }
        offset += n;
        file.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
        return !!file;
    }

    uint64_t remaining() const {
        return offset <= size ? size - offset : 0;
    }
};

uint32_t u32le(const unsigned char * p) {
    uint32_t v = 0;
    for (size_t i = 0; i < 4; ++i) v |= static_cast<uint32_t>(p[i]) << (8*i);
    return v;
}

uint64_t u64le(const unsigned char * p) {
    uint64_t v = 0;
    for (size_t i = 0; i < 8; ++i) v |= static_cast<uint64_t>(p[i]) << (8*i);
    return v;
}

bool read_u32(reader & r, uint32_t & v) {
    unsigned char b[4];
    if (!r.read(b, sizeof(b))) return false;
    v = u32le(b);
    return true;
}

bool read_u64(reader & r, uint64_t & v) {
    unsigned char b[8];
    if (!r.read(b, sizeof(b))) return false;
    v = u64le(b);
    return true;
}

common_tree_draft_gguf_tensor_result fail(common_tree_draft_gguf_tensor_error error, uint64_t offset, const char * message) {
    common_tree_draft_gguf_tensor_result result;
    result.error = error;
    result.error_offset = offset;
    result.message = message;
    return result;
}

bool read_string(reader & r, std::string & value) {
    uint64_t n = 0;
    if (!read_u64(r, n) || n > r.remaining() || n > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) return false;
    try {
        value.resize(static_cast<size_t>(n));
    } catch (const std::bad_alloc &) {
        return false;
    } catch (const std::length_error &) {
        return false;
    }
    return n == 0 || r.read(value.data(), n);
}

size_t gguf_scalar_size(gguf_type type) {
    switch (type) {
        case GGUF_TYPE_UINT8: return 1;
        case GGUF_TYPE_INT8: return 1;
        case GGUF_TYPE_UINT16: return 2;
        case GGUF_TYPE_INT16: return 2;
        case GGUF_TYPE_UINT32: return 4;
        case GGUF_TYPE_INT32: return 4;
        case GGUF_TYPE_FLOAT32: return 4;
        case GGUF_TYPE_BOOL: return 1;
        case GGUF_TYPE_UINT64: return 8;
        case GGUF_TYPE_INT64: return 8;
        case GGUF_TYPE_FLOAT64: return 8;
        case GGUF_TYPE_STRING:
        case GGUF_TYPE_ARRAY:
        case GGUF_TYPE_COUNT: return 0;
    }
    return 0;
}

bool skip_string(reader & r) {
    uint64_t n = 0;
    return read_u64(r, n) && r.skip(n);
}

bool skip_metadata_value(reader & r, gguf_type type) {
    if (type == GGUF_TYPE_STRING) return skip_string(r);
    if (type == GGUF_TYPE_ARRAY) {
        uint32_t elem_raw = 0;
        uint64_t count = 0;
        if (!read_u32(r, elem_raw) || !read_u64(r, count) || elem_raw >= GGUF_TYPE_COUNT) return false;
        const auto elem = static_cast<gguf_type>(elem_raw);
        if (elem == GGUF_TYPE_ARRAY) return false;
        if (elem == GGUF_TYPE_STRING) {
            for (uint64_t i = 0; i < count; ++i) if (!skip_string(r)) return false;
            return true;
        }
        const size_t s = gguf_scalar_size(elem);
        return s != 0 && count <= std::numeric_limits<uint64_t>::max()/s && r.skip(count*s);
    }
    const size_t s = gguf_scalar_size(type);
    return s != 0 && r.skip(s);
}

bool skip_metadata(reader & r, uint64_t n_kv, uint32_t & alignment) {
    std::unordered_set<std::string> keys;
    for (uint64_t i = 0; i < n_kv; ++i) {
        std::string key;
        uint32_t type_raw = 0;
        if (!read_string(r, key) || key.empty() || !keys.insert(key).second ||
                !read_u32(r, type_raw) || type_raw >= GGUF_TYPE_COUNT) return false;
        const auto type = static_cast<gguf_type>(type_raw);
        if (key == GGUF_KEY_GENERAL_ALIGNMENT && type == GGUF_TYPE_UINT32) {
            uint32_t v = 0;
            if (!read_u32(r, v) || v == 0 || (v & (v - 1)) != 0) return false;
            alignment = v;
        } else if (!skip_metadata_value(r, type)) {
            return false;
        }
    }
    return true;
}

bool checked_mul(uint64_t a, uint64_t b, uint64_t & out) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max()/a) return false;
    out = a*b;
    return true;
}

bool checked_add(uint64_t a, uint64_t b, uint64_t & out) {
    if (b > std::numeric_limits<uint64_t>::max() - a) return false;
    out = a+b;
    return true;
}

bool align_up(uint64_t value, uint32_t alignment, uint64_t & out) {
    const uint64_t mask = static_cast<uint64_t>(alignment - 1);
    if (value > std::numeric_limits<uint64_t>::max() - mask) return false;
    out = (value + mask) & ~mask;
    return true;
}

}

common_tree_draft_gguf_tensor_result common_tree_draft_gguf_tensor_directory_parse(
        const common_tree_draft_model_source & source,
        size_t path_index) {
    const auto header = common_tree_draft_gguf_header_parse(source, path_index);
    if (!header) {
        auto result = fail(header.error == COMMON_TREE_DRAFT_GGUF_HEADER_INVALID_SOURCE ?
                COMMON_TREE_DRAFT_GGUF_TENSOR_INVALID_SOURCE : COMMON_TREE_DRAFT_GGUF_TENSOR_HEADER_ERROR,
                header.error_offset, header.message.c_str());
        result.header_error = header.error;
        return result;
    }

    const fs::path path(source.paths[path_index]);
    std::error_code ec;
    const uintmax_t native_size = fs::file_size(path, ec);
    if (ec || native_size > std::numeric_limits<uint64_t>::max()) {
        return fail(COMMON_TREE_DRAFT_GGUF_TENSOR_OPEN_FAILED, 0, "failed to obtain GGUF file size");
    }
    const uint64_t file_size = static_cast<uint64_t>(native_size);
    reader r(path, file_size, header.header.header_end);
    if (!r.file) return fail(COMMON_TREE_DRAFT_GGUF_TENSOR_OPEN_FAILED, 0, "failed to open GGUF file");

    uint32_t alignment = GGUF_DEFAULT_ALIGNMENT;
    if (!skip_metadata(r, header.header.n_kv, alignment)) {
        return fail(COMMON_TREE_DRAFT_GGUF_TENSOR_METADATA_INVALID, r.offset, "failed to skip GGUF metadata safely");
    }

    common_tree_draft_gguf_tensor_result result;
    result.directory.alignment = alignment;
    std::unordered_set<std::string> names;
    if (header.header.n_tensors > r.remaining()/24) {
        return fail(COMMON_TREE_DRAFT_GGUF_TENSOR_TRUNCATED, r.offset, "tensor count exceeds remaining directory bytes");
    }
    result.directory.tensors.reserve(static_cast<size_t>(header.header.n_tensors));

    for (uint64_t i = 0; i < header.header.n_tensors; ++i) {
        common_tree_draft_gguf_tensor_descriptor d;
        if (!read_string(r, d.name)) return fail(COMMON_TREE_DRAFT_GGUF_TENSOR_TRUNCATED, r.offset, "truncated tensor name");
        if (d.name.empty()) return fail(COMMON_TREE_DRAFT_GGUF_TENSOR_EMPTY_NAME, r.offset, "tensor name is empty");
        if (!names.insert(d.name).second) return fail(COMMON_TREE_DRAFT_GGUF_TENSOR_DUPLICATE_NAME, r.offset, "duplicate tensor name");

        uint32_t rank = 0;
        if (!read_u32(r, rank)) return fail(COMMON_TREE_DRAFT_GGUF_TENSOR_TRUNCATED, r.offset, "truncated tensor rank");
        if (rank < 1 || rank > GGML_MAX_DIMS) return fail(COMMON_TREE_DRAFT_GGUF_TENSOR_INVALID_RANK, r.offset, "invalid tensor rank");
        d.dims.reserve(rank);
        uint64_t nelements = 1;
        for (uint32_t j = 0; j < rank; ++j) {
            uint64_t raw_dim = 0;
            if (!read_u64(r, raw_dim)) return fail(COMMON_TREE_DRAFT_GGUF_TENSOR_TRUNCATED, r.offset, "truncated tensor dimension");
            if (raw_dim == 0 || raw_dim > static_cast<uint64_t>(INT64_MAX)) return fail(COMMON_TREE_DRAFT_GGUF_TENSOR_INVALID_DIMENSION, r.offset, "invalid tensor dimension");
            if (!checked_mul(nelements, raw_dim, nelements)) return fail(COMMON_TREE_DRAFT_GGUF_TENSOR_SIZE_OVERFLOW, r.offset, "tensor element count overflows");
            d.dims.push_back(static_cast<int64_t>(raw_dim));
        }

        uint32_t type_raw = 0;
        if (!read_u32(r, type_raw) || type_raw >= GGML_TYPE_COUNT) return fail(COMMON_TREE_DRAFT_GGUF_TENSOR_INVALID_TYPE, r.offset, "invalid tensor type");
        d.type = static_cast<ggml_type>(type_raw);
        const int64_t block = ggml_blck_size(d.type);
        const size_t type_size = ggml_type_size(d.type);
        if (block <= 0 || type_size == 0 || d.dims[0] % block != 0) return fail(COMMON_TREE_DRAFT_GGUF_TENSOR_INVALID_TYPE, r.offset, "tensor type is incompatible with first dimension");

        uint64_t blocks = nelements/static_cast<uint64_t>(block);
        if (!checked_mul(blocks, type_size, d.storage_size)) return fail(COMMON_TREE_DRAFT_GGUF_TENSOR_SIZE_OVERFLOW, r.offset, "tensor storage size overflows");
        if (!read_u64(r, d.offset)) return fail(COMMON_TREE_DRAFT_GGUF_TENSOR_TRUNCATED, r.offset, "truncated tensor data offset");
        result.directory.tensors.push_back(std::move(d));
    }

    result.directory.directory_end = r.offset;
    if (!align_up(r.offset, alignment, result.directory.data_base) || result.directory.data_base > file_size) {
        return fail(COMMON_TREE_DRAFT_GGUF_TENSOR_SPAN_OUT_OF_BOUNDS, r.offset, "tensor data base is outside file bounds");
    }

    for (const auto & d : result.directory.tensors) {
        uint64_t start = 0;
        uint64_t end = 0;
        if (!checked_add(result.directory.data_base, d.offset, start) ||
                !checked_add(start, d.storage_size, end) || end > file_size) {
            return fail(COMMON_TREE_DRAFT_GGUF_TENSOR_SPAN_OUT_OF_BOUNDS, d.offset, "tensor storage span exceeds file bounds");
        }
    }
    return result;
}

const char * common_tree_draft_gguf_tensor_error_name(common_tree_draft_gguf_tensor_error error) {
    switch (error) {
        case COMMON_TREE_DRAFT_GGUF_TENSOR_OK: return "ok";
        case COMMON_TREE_DRAFT_GGUF_TENSOR_INVALID_SOURCE: return "invalid_source";
        case COMMON_TREE_DRAFT_GGUF_TENSOR_HEADER_ERROR: return "header_error";
        case COMMON_TREE_DRAFT_GGUF_TENSOR_OPEN_FAILED: return "open_failed";
        case COMMON_TREE_DRAFT_GGUF_TENSOR_TRUNCATED: return "truncated";
        case COMMON_TREE_DRAFT_GGUF_TENSOR_METADATA_INVALID: return "metadata_invalid";
        case COMMON_TREE_DRAFT_GGUF_TENSOR_EMPTY_NAME: return "empty_name";
        case COMMON_TREE_DRAFT_GGUF_TENSOR_DUPLICATE_NAME: return "duplicate_name";
        case COMMON_TREE_DRAFT_GGUF_TENSOR_INVALID_RANK: return "invalid_rank";
        case COMMON_TREE_DRAFT_GGUF_TENSOR_INVALID_DIMENSION: return "invalid_dimension";
        case COMMON_TREE_DRAFT_GGUF_TENSOR_INVALID_TYPE: return "invalid_type";
        case COMMON_TREE_DRAFT_GGUF_TENSOR_SIZE_OVERFLOW: return "size_overflow";
        case COMMON_TREE_DRAFT_GGUF_TENSOR_SPAN_OUT_OF_BOUNDS: return "span_out_of_bounds";
    }
    return "unknown";
}
