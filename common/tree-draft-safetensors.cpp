#include "tree-draft-safetensors.h"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <limits>
#include <unordered_set>
#include <utility>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

uint64_t read_u64_le(const unsigned char * bytes) {
    uint64_t value = 0;
    for (size_t i = 0; i < sizeof(value); ++i) {
        value |= static_cast<uint64_t>(bytes[i]) << (8 * i);
    }
    return value;
}

common_tree_draft_safetensors_result fail(
        common_tree_draft_safetensors_error error,
        uint64_t offset,
        const char * message) {
    common_tree_draft_safetensors_result result;
    result.error = error;
    result.error_offset = offset;
    result.message = message;
    return result;
}

bool json_u64(const json & value, uint64_t & result) {
    if (value.is_number_unsigned()) {
        result = value.get<uint64_t>();
        return true;
    }
    if (value.is_number_integer()) {
        const int64_t signed_value = value.get<int64_t>();
        if (signed_value < 0) {
            return false;
        }
        result = static_cast<uint64_t>(signed_value);
        return true;
    }
    return false;
}

bool dtype_size(const std::string & dtype, uint64_t & result) {
    if (dtype == "BOOL" || dtype == "U8" || dtype == "I8" || dtype == "F8_E4M3FN" || dtype == "F8_E5M2") {
        result = 1;
        return true;
    }
    if (dtype == "U16" || dtype == "I16" || dtype == "F16" || dtype == "BF16") {
        result = 2;
        return true;
    }
    if (dtype == "U32" || dtype == "I32" || dtype == "F32") {
        result = 4;
        return true;
    }
    if (dtype == "U64" || dtype == "I64" || dtype == "F64") {
        result = 8;
        return true;
    }
    return false;
}

bool checked_product(const std::vector<uint64_t> & values, uint64_t & product) {
    product = 1;
    for (const uint64_t value : values) {
        if (value != 0 && product > std::numeric_limits<uint64_t>::max() / value) {
            return false;
        }
        product *= value;
    }
    return true;
}

}

common_tree_draft_safetensors_result common_tree_draft_safetensors_parse(
        const common_tree_draft_model_source & source,
        size_t path_index) {
    if (source.format != COMMON_TREE_DRAFT_MODEL_FORMAT_SAFETENSORS ||
            source.loader != COMMON_TREE_DRAFT_MODEL_LOADER_SAFETENSORS ||
            path_index >= source.paths.size()) {
        return fail(COMMON_TREE_DRAFT_SAFETENSORS_INVALID_SOURCE, 0, "source is not a selected Safetensors input");
    }

    const fs::path path(source.paths[path_index]);
    std::error_code ec;
    const uintmax_t native_size = fs::file_size(path, ec);
    if (ec || native_size > std::numeric_limits<uint64_t>::max()) {
        return fail(COMMON_TREE_DRAFT_SAFETENSORS_OPEN_FAILED, 0, "failed to obtain Safetensors file size");
    }
    const uint64_t file_size = static_cast<uint64_t>(native_size);
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return fail(COMMON_TREE_DRAFT_SAFETENSORS_OPEN_FAILED, 0, "failed to open Safetensors file");
    }
    if (file_size < sizeof(uint64_t)) {
        return fail(COMMON_TREE_DRAFT_SAFETENSORS_TRUNCATED, file_size, "Safetensors header length is truncated");
    }

    std::array<unsigned char, sizeof(uint64_t)> length_bytes{};
    file.read(reinterpret_cast<char *>(length_bytes.data()), static_cast<std::streamsize>(length_bytes.size()));
    if (static_cast<size_t>(file.gcount()) != length_bytes.size()) {
        return fail(COMMON_TREE_DRAFT_SAFETENSORS_TRUNCATED, static_cast<uint64_t>(file.gcount()), "Safetensors header length is truncated");
    }

    const uint64_t header_size = read_u64_le(length_bytes.data());
    if (header_size > file_size - sizeof(uint64_t)) {
        return fail(COMMON_TREE_DRAFT_SAFETENSORS_TRUNCATED, sizeof(uint64_t), "Safetensors header is truncated");
    }
    if (header_size > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
            header_size > static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max())) {
        return fail(COMMON_TREE_DRAFT_SAFETENSORS_HEADER_TOO_LARGE, sizeof(uint64_t), "Safetensors header length is not representable");
    }

    std::string header(static_cast<size_t>(header_size), '\0');
    if (header_size > 0) {
        file.read(header.data(), static_cast<std::streamsize>(header.size()));
        if (static_cast<size_t>(file.gcount()) != header.size()) {
            return fail(COMMON_TREE_DRAFT_SAFETENSORS_TRUNCATED, sizeof(uint64_t), "Safetensors header is truncated");
        }
    }

    bool duplicate_tensor = false;
    std::unordered_set<std::string> tensor_names;
    json root;
    try {
        root = json::parse(header, [&duplicate_tensor, &tensor_names](int depth, json::parse_event_t event, json & parsed) {
            if (depth == 1 && event == json::parse_event_t::key) {
                const std::string key = parsed.get<std::string>();
                if (key != "__metadata__" && !tensor_names.insert(key).second) {
                    duplicate_tensor = true;
                }
            }
            return true;
        });
    } catch (const json::exception &) {
        return fail(COMMON_TREE_DRAFT_SAFETENSORS_INVALID_JSON, sizeof(uint64_t), "invalid Safetensors header JSON");
    }
    if (!root.is_object()) {
        return fail(COMMON_TREE_DRAFT_SAFETENSORS_INVALID_HEADER, sizeof(uint64_t), "Safetensors header must be an object");
    }
    if (duplicate_tensor) {
        return fail(COMMON_TREE_DRAFT_SAFETENSORS_DUPLICATE_TENSOR, sizeof(uint64_t), "duplicate Safetensors tensor name");
    }

    common_tree_draft_safetensors_result result;
    result.table.header_size = header_size;
    result.table.payload_size = file_size - sizeof(uint64_t) - header_size;

    for (const auto & item : root.items()) {
        if (item.key() == "__metadata__") {
            if (!item.value().is_object()) {
                return fail(COMMON_TREE_DRAFT_SAFETENSORS_INVALID_HEADER, sizeof(uint64_t), "Safetensors metadata must be an object");
            }
            for (const auto & metadata : item.value().items()) {
                if (!metadata.value().is_string()) {
                    return fail(COMMON_TREE_DRAFT_SAFETENSORS_INVALID_HEADER, sizeof(uint64_t), "Safetensors metadata values must be strings");
                }
                result.table.metadata.emplace(metadata.key(), metadata.value().get<std::string>());
            }
            continue;
        }

        if (!item.value().is_object()) {
            return fail(COMMON_TREE_DRAFT_SAFETENSORS_INVALID_HEADER, sizeof(uint64_t), "Safetensors tensor descriptor must be an object");
        }
        const auto dtype = item.value().find("dtype");
        const auto shape = item.value().find("shape");
        const auto offsets = item.value().find("data_offsets");
        if (dtype == item.value().end() || !dtype->is_string() ||
                shape == item.value().end() || !shape->is_array() ||
                offsets == item.value().end() || !offsets->is_array() || offsets->size() != 2) {
            return fail(COMMON_TREE_DRAFT_SAFETENSORS_INVALID_HEADER, sizeof(uint64_t), "invalid Safetensors tensor descriptor");
        }

        common_tree_draft_safetensors_tensor tensor;
        tensor.name = item.key();
        tensor.dtype = dtype->get<std::string>();
        uint64_t item_size = 0;
        if (!dtype_size(tensor.dtype, item_size)) {
            return fail(COMMON_TREE_DRAFT_SAFETENSORS_UNSUPPORTED_DTYPE, sizeof(uint64_t), "unsupported Safetensors dtype");
        }
        for (const auto & dimension : *shape) {
            uint64_t value = 0;
            if (!json_u64(dimension, value)) {
                return fail(COMMON_TREE_DRAFT_SAFETENSORS_INVALID_SHAPE, sizeof(uint64_t), "invalid Safetensors tensor shape");
            }
            tensor.shape.push_back(value);
        }
        if (!json_u64((*offsets)[0], tensor.offset_begin) ||
                !json_u64((*offsets)[1], tensor.offset_end) ||
                tensor.offset_begin > tensor.offset_end ||
                tensor.offset_end > result.table.payload_size) {
            return fail(COMMON_TREE_DRAFT_SAFETENSORS_INVALID_OFFSETS, sizeof(uint64_t), "invalid Safetensors tensor offsets");
        }

        uint64_t element_count = 0;
        if (!checked_product(tensor.shape, element_count) ||
                element_count > std::numeric_limits<uint64_t>::max() / item_size ||
                element_count * item_size != tensor.offset_end - tensor.offset_begin) {
            return fail(COMMON_TREE_DRAFT_SAFETENSORS_SIZE_MISMATCH, sizeof(uint64_t), "Safetensors tensor byte size does not match shape and dtype");
        }
        result.table.tensors.push_back(std::move(tensor));
    }

    std::vector<const common_tree_draft_safetensors_tensor *> ranges;
    ranges.reserve(result.table.tensors.size());
    for (const auto & tensor : result.table.tensors) {
        ranges.push_back(&tensor);
    }
    std::sort(ranges.begin(), ranges.end(), [](const auto * lhs, const auto * rhs) {
        if (lhs->offset_begin != rhs->offset_begin) {
            return lhs->offset_begin < rhs->offset_begin;
        }
        return lhs->offset_end < rhs->offset_end;
    });
    for (size_t i = 1; i < ranges.size(); ++i) {
        if (ranges[i - 1]->offset_end > ranges[i]->offset_begin) {
            return fail(COMMON_TREE_DRAFT_SAFETENSORS_OVERLAPPING_TENSORS, sizeof(uint64_t), "Safetensors tensor ranges overlap");
        }
    }

    return result;
}

const char * common_tree_draft_safetensors_error_name(common_tree_draft_safetensors_error error) {
    switch (error) {
        case COMMON_TREE_DRAFT_SAFETENSORS_OK:                  return "ok";
        case COMMON_TREE_DRAFT_SAFETENSORS_INVALID_SOURCE:      return "invalid_source";
        case COMMON_TREE_DRAFT_SAFETENSORS_OPEN_FAILED:         return "open_failed";
        case COMMON_TREE_DRAFT_SAFETENSORS_TRUNCATED:           return "truncated";
        case COMMON_TREE_DRAFT_SAFETENSORS_HEADER_TOO_LARGE:    return "header_too_large";
        case COMMON_TREE_DRAFT_SAFETENSORS_INVALID_JSON:        return "invalid_json";
        case COMMON_TREE_DRAFT_SAFETENSORS_INVALID_HEADER:      return "invalid_header";
        case COMMON_TREE_DRAFT_SAFETENSORS_DUPLICATE_TENSOR:    return "duplicate_tensor";
        case COMMON_TREE_DRAFT_SAFETENSORS_UNSUPPORTED_DTYPE:   return "unsupported_dtype";
        case COMMON_TREE_DRAFT_SAFETENSORS_INVALID_SHAPE:       return "invalid_shape";
        case COMMON_TREE_DRAFT_SAFETENSORS_INVALID_OFFSETS:     return "invalid_offsets";
        case COMMON_TREE_DRAFT_SAFETENSORS_SIZE_MISMATCH:       return "size_mismatch";
        case COMMON_TREE_DRAFT_SAFETENSORS_OVERLAPPING_TENSORS: return "overlapping_tensors";
    }
    return "unknown";
}
