#include "tree-draft-gguf-shards.h"

#include <algorithm>
#include <cstring>
#include <unordered_set>

namespace {

common_tree_draft_gguf_shard_result fail(
        common_tree_draft_gguf_shard_error error,
        size_t path_index,
        const char * message) {
    common_tree_draft_gguf_shard_result result;
    result.error = error;
    result.source_path_index = path_index;
    result.message = message;
    return result;
}

bool read_u16(const common_tree_draft_gguf_metadata_entry & entry, uint16_t & value) {
    if (entry.value.is_array || entry.value.type != GGUF_TYPE_UINT16 || entry.value.raw.size() != sizeof(uint16_t)) {
        return false;
    }
    value = static_cast<uint16_t>(entry.value.raw[0]) |
        static_cast<uint16_t>(static_cast<uint16_t>(entry.value.raw[1]) << 8);
    return true;
}

bool read_i32(const common_tree_draft_gguf_metadata_entry & entry, int32_t & value) {
    if (entry.value.is_array || entry.value.type != GGUF_TYPE_INT32 || entry.value.raw.size() != sizeof(int32_t)) {
        return false;
    }
    const uint32_t raw = static_cast<uint32_t>(entry.value.raw[0]) |
        (static_cast<uint32_t>(entry.value.raw[1]) << 8) |
        (static_cast<uint32_t>(entry.value.raw[2]) << 16) |
        (static_cast<uint32_t>(entry.value.raw[3]) << 24);
    std::memcpy(&value, &raw, sizeof(value));
    return true;
}

bool is_split_key(const std::string & key) {
    return key == "split.no" || key == "split.count" || key == "split.tensors.count";
}

bool same_value(
        const common_tree_draft_gguf_metadata_value & a,
        const common_tree_draft_gguf_metadata_value & b) {
    return a.type == b.type && a.is_array == b.is_array && a.array_type == b.array_type &&
        a.raw == b.raw && a.strings == b.strings;
}

bool same_model_identity(
        const common_tree_draft_gguf_metadata_table & a,
        const common_tree_draft_gguf_metadata_table & b) {
    std::vector<const common_tree_draft_gguf_metadata_entry *> av;
    std::vector<const common_tree_draft_gguf_metadata_entry *> bv;
    for (const auto & e : a.entries) if (!is_split_key(e.key)) av.push_back(&e);
    for (const auto & e : b.entries) if (!is_split_key(e.key)) bv.push_back(&e);
    if (av.size() != bv.size()) return false;
    auto cmp = [](const auto * lhs, const auto * rhs) { return lhs->key < rhs->key; };
    std::sort(av.begin(), av.end(), cmp);
    std::sort(bv.begin(), bv.end(), cmp);
    for (size_t i = 0; i < av.size(); ++i) {
        if (av[i]->key != bv[i]->key || !same_value(av[i]->value, bv[i]->value)) return false;
    }
    return true;
}

struct parsed_shard {
    size_t source_path_index = 0;
    uint16_t index = 0;
    uint16_t count = 0;
    uint32_t tensor_count = 0;
    common_tree_draft_gguf_metadata_table metadata;
    common_tree_draft_gguf_tensor_directory directory;
};

}

common_tree_draft_gguf_shard_result common_tree_draft_gguf_resolve_shards(
        const common_tree_draft_model_source & source) {
    if (source.format != COMMON_TREE_DRAFT_MODEL_FORMAT_GGUF ||
            source.loader != COMMON_TREE_DRAFT_MODEL_LOADER_GGUF || source.paths.empty()) {
        return fail(COMMON_TREE_DRAFT_GGUF_SHARD_INVALID_SOURCE, 0, "source is not a GGUF shard set");
    }

    std::vector<parsed_shard> shards;
    shards.reserve(source.paths.size());
    for (size_t path_index = 0; path_index < source.paths.size(); ++path_index) {
        const auto metadata = common_tree_draft_gguf_metadata_parse(source, path_index);
        if (!metadata) return fail(COMMON_TREE_DRAFT_GGUF_SHARD_METADATA_ERROR, path_index, metadata.message.c_str());
        const auto directory = common_tree_draft_gguf_tensor_directory_parse(source, path_index);
        if (!directory) return fail(COMMON_TREE_DRAFT_GGUF_SHARD_TENSOR_ERROR, path_index, directory.message.c_str());

        const auto split_no = common_tree_draft_gguf_metadata_find(metadata.table, "split.no");
        const auto split_count = common_tree_draft_gguf_metadata_find(metadata.table, "split.count");
        const auto split_tensors = common_tree_draft_gguf_metadata_find(metadata.table, "split.tensors.count");
        if (!split_no || !split_count || !split_tensors) {
            return fail(COMMON_TREE_DRAFT_GGUF_SHARD_MISSING_SPLIT_METADATA, path_index, "required split metadata is missing");
        }

        parsed_shard shard;
        shard.source_path_index = path_index;
        int32_t total_tensors = 0;
        if (!read_u16(*split_no.entry, shard.index) || !read_u16(*split_count.entry, shard.count) ||
                !read_i32(*split_tensors.entry, total_tensors) || shard.count == 0 || shard.index >= shard.count || total_tensors < 0) {
            return fail(COMMON_TREE_DRAFT_GGUF_SHARD_MALFORMED_SPLIT_METADATA, path_index, "split metadata has invalid type or value");
        }
        shard.tensor_count = static_cast<uint32_t>(total_tensors);
        shard.metadata = metadata.table;
        shard.directory = directory.directory;
        shards.push_back(std::move(shard));
    }

    const uint16_t expected_count = shards.front().count;
    const uint32_t expected_tensors = shards.front().tensor_count;
    if (expected_count != shards.size()) {
        return fail(COMMON_TREE_DRAFT_GGUF_SHARD_MISSING_INDEX, 0, "split count does not match number of supplied shards");
    }
    for (size_t i = 1; i < shards.size(); ++i) {
        if (shards[i].count != expected_count || shards[i].tensor_count != expected_tensors) {
            return fail(COMMON_TREE_DRAFT_GGUF_SHARD_INCONSISTENT_SPLIT_COUNT, shards[i].source_path_index, "split metadata is inconsistent across shards");
        }
        if (!same_model_identity(shards.front().metadata, shards[i].metadata)) {
            return fail(COMMON_TREE_DRAFT_GGUF_SHARD_FOREIGN_MODEL, shards[i].source_path_index, "non-split model metadata differs across shards");
        }
    }

    std::sort(shards.begin(), shards.end(), [](const parsed_shard & a, const parsed_shard & b) {
        return a.index < b.index;
    });
    for (size_t i = 0; i < shards.size(); ++i) {
        if (i > 0 && shards[i - 1].index == shards[i].index) {
            return fail(COMMON_TREE_DRAFT_GGUF_SHARD_DUPLICATE_INDEX, shards[i].source_path_index, "duplicate split index");
        }
        if (shards[i].index != i) {
            return fail(COMMON_TREE_DRAFT_GGUF_SHARD_MISSING_INDEX, shards[i].source_path_index, "split index set is incomplete");
        }
    }

    common_tree_draft_gguf_shard_result result;
    result.set.split_count = expected_count;
    result.set.expected_tensor_count = expected_tensors;
    std::unordered_set<std::string> tensor_names;
    for (const auto & shard : shards) {
        result.set.source_path_indices.push_back(shard.source_path_index);
        result.set.source_paths.push_back(source.paths[shard.source_path_index]);
        for (const auto & descriptor : shard.directory.tensors) {
            if (!tensor_names.insert(descriptor.name).second) {
                return fail(COMMON_TREE_DRAFT_GGUF_SHARD_DUPLICATE_TENSOR, shard.source_path_index, "tensor name is duplicated across shards");
            }
            result.set.tensors.push_back({ descriptor, shard.index, shard.source_path_index, source.paths[shard.source_path_index] });
        }
    }
    if (result.set.tensors.size() != expected_tensors) {
        return fail(COMMON_TREE_DRAFT_GGUF_SHARD_TENSOR_COUNT_MISMATCH, 0, "logical tensor count does not match split.tensors.count");
    }
    return result;
}

const char * common_tree_draft_gguf_shard_error_name(common_tree_draft_gguf_shard_error error) {
    switch (error) {
        case COMMON_TREE_DRAFT_GGUF_SHARD_OK: return "ok";
        case COMMON_TREE_DRAFT_GGUF_SHARD_INVALID_SOURCE: return "invalid_source";
        case COMMON_TREE_DRAFT_GGUF_SHARD_METADATA_ERROR: return "metadata_error";
        case COMMON_TREE_DRAFT_GGUF_SHARD_TENSOR_ERROR: return "tensor_error";
        case COMMON_TREE_DRAFT_GGUF_SHARD_MISSING_SPLIT_METADATA: return "missing_split_metadata";
        case COMMON_TREE_DRAFT_GGUF_SHARD_MALFORMED_SPLIT_METADATA: return "malformed_split_metadata";
        case COMMON_TREE_DRAFT_GGUF_SHARD_INCONSISTENT_SPLIT_COUNT: return "inconsistent_split_count";
        case COMMON_TREE_DRAFT_GGUF_SHARD_DUPLICATE_INDEX: return "duplicate_index";
        case COMMON_TREE_DRAFT_GGUF_SHARD_MISSING_INDEX: return "missing_index";
        case COMMON_TREE_DRAFT_GGUF_SHARD_FOREIGN_MODEL: return "foreign_model";
        case COMMON_TREE_DRAFT_GGUF_SHARD_DUPLICATE_TENSOR: return "duplicate_tensor";
        case COMMON_TREE_DRAFT_GGUF_SHARD_TENSOR_COUNT_MISMATCH: return "tensor_count_mismatch";
    }
    return "unknown";
}
