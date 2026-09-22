#include "tree-draft-safetensors-index.h"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <unordered_map>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

common_tree_draft_safetensors_index_result fail(
        common_tree_draft_safetensors_index_error error,
        const char * message) {
    common_tree_draft_safetensors_index_result result;
    result.error = error;
    result.message = message;
    return result;
}

bool find_index_path(const common_tree_draft_model_source & source, fs::path & index_path) {
    bool found = false;
    for (const auto & evidence : source.evidence) {
        if (evidence.signal != COMMON_TREE_DRAFT_MODEL_PROBE_SAFETENSORS_INDEX) {
            continue;
        }
        if (found) {
            return false;
        }
        index_path = fs::path(evidence.path);
        found = true;
    }
    return found;
}

std::string normalized(const fs::path & path) {
    return path.lexically_normal().generic_string();
}

}

common_tree_draft_safetensors_index_result common_tree_draft_safetensors_index_resolve(
        const common_tree_draft_model_source & source) {
    fs::path index_path;
    if (!find_index_path(source, index_path)) {
        return fail(COMMON_TREE_DRAFT_SAFETENSORS_INDEX_INVALID_SOURCE, "source does not identify exactly one Safetensors index");
    }

    std::ifstream file(index_path, std::ios::binary);
    if (!file) {
        return fail(COMMON_TREE_DRAFT_SAFETENSORS_INDEX_MISSING_INDEX, "Safetensors index file is missing");
    }

    bool duplicate_weight = false;
    std::set<std::string> weight_names;
    bool next_object_is_weight_map = false;
    bool in_weight_map = false;
    int weight_map_depth = -1;
    json root;
    try {
        root = json::parse(file, [&](int depth, json::parse_event_t event, json & parsed) {
            if (event == json::parse_event_t::key && depth == 1) {
                next_object_is_weight_map = parsed.get<std::string>() == "weight_map";
            } else if (event == json::parse_event_t::object_start && next_object_is_weight_map) {
                in_weight_map = true;
                weight_map_depth = depth;
                next_object_is_weight_map = false;
            } else if (event == json::parse_event_t::object_end && in_weight_map && depth == weight_map_depth) {
                in_weight_map = false;
                weight_map_depth = -1;
            } else if (in_weight_map && event == json::parse_event_t::key && depth == weight_map_depth + 1) {
                const std::string key = parsed.get<std::string>();
                if (!weight_names.insert(key).second) {
                    duplicate_weight = true;
                }
            }
            return true;
        });
    } catch (const json::exception &) {
        return fail(COMMON_TREE_DRAFT_SAFETENSORS_INDEX_INVALID_JSON, "invalid Safetensors index JSON");
    }
    if (!root.is_object()) {
        return fail(COMMON_TREE_DRAFT_SAFETENSORS_INDEX_INVALID_JSON, "Safetensors index root must be an object");
    }
    const auto weight_map_it = root.find("weight_map");
    if (weight_map_it == root.end() || !weight_map_it->is_object() || weight_map_it->empty()) {
        return fail(COMMON_TREE_DRAFT_SAFETENSORS_INDEX_INVALID_WEIGHT_MAP, "Safetensors index weight_map must be a non-empty object");
    }
    if (duplicate_weight) {
        return fail(COMMON_TREE_DRAFT_SAFETENSORS_INDEX_DUPLICATE_TENSOR, "duplicate tensor name in Safetensors weight_map");
    }

    std::map<std::string, std::string> weight_map;
    std::set<std::string> referenced_shards;
    for (const auto & item : weight_map_it->items()) {
        if (item.key().empty() || !item.value().is_string()) {
            return fail(COMMON_TREE_DRAFT_SAFETENSORS_INDEX_INVALID_WEIGHT_MAP, "Safetensors weight_map entry is invalid");
        }
        const std::string shard_name = item.value().get<std::string>();
        if (shard_name.empty()) {
            return fail(COMMON_TREE_DRAFT_SAFETENSORS_INDEX_INVALID_WEIGHT_MAP, "Safetensors weight_map shard name is empty");
        }
        weight_map.emplace(item.key(), shard_name);
        referenced_shards.insert(shard_name);
    }

    std::unordered_map<std::string, size_t> source_index_by_path;
    for (size_t i = 0; i < source.paths.size(); ++i) {
        source_index_by_path.emplace(normalized(fs::path(source.paths[i])), i);
    }

    struct shard_data {
        size_t source_path_index = 0;
        std::string path;
        common_tree_draft_safetensors_table table;
    };
    std::map<std::string, shard_data> shards;
    for (const auto & shard_name : referenced_shards) {
        const fs::path shard_path = (index_path.parent_path() / fs::path(shard_name)).lexically_normal();
        const std::string shard_path_string = normalized(shard_path);
        const auto source_it = source_index_by_path.find(shard_path_string);
        if (source_it == source_index_by_path.end() || !fs::is_regular_file(shard_path)) {
            return fail(COMMON_TREE_DRAFT_SAFETENSORS_INDEX_MISSING_SHARD, "referenced Safetensors shard is missing");
        }
        const auto shard_source = common_tree_draft_model_source_probe(shard_path_string);
        if (shard_source.format != COMMON_TREE_DRAFT_MODEL_FORMAT_SAFETENSORS ||
                shard_source.loader != COMMON_TREE_DRAFT_MODEL_LOADER_SAFETENSORS) {
            return fail(COMMON_TREE_DRAFT_SAFETENSORS_INDEX_SHARD_PARSE_ERROR, "referenced shard is not a valid Safetensors source");
        }
        const auto parsed = common_tree_draft_safetensors_parse(shard_source);
        if (!parsed) {
            return fail(COMMON_TREE_DRAFT_SAFETENSORS_INDEX_SHARD_PARSE_ERROR, parsed.message.c_str());
        }
        shards.emplace(shard_name, shard_data { source_it->second, shard_path_string, parsed.table });
    }

    for (const auto & shard_pair : shards) {
        const std::string & shard_name = shard_pair.first;
        const auto & shard = shard_pair.second;
        for (const auto & tensor : shard.table.tensors) {
            const auto index_it = weight_map.find(tensor.name);
            if (index_it == weight_map.end() || index_it->second != shard_name) {
                return fail(COMMON_TREE_DRAFT_SAFETENSORS_INDEX_TENSOR_CONFLICT, "shard tensor conflicts with Safetensors weight_map");
            }
        }
    }

    common_tree_draft_safetensors_index_result result;
    result.directory.index_path = normalized(index_path);
    for (const auto & shard_pair : shards) {
        result.directory.shard_paths.push_back(shard_pair.second.path);
    }

    for (const auto & weight : weight_map) {
        const auto shard_it = shards.find(weight.second);
        if (shard_it == shards.end()) {
            return fail(COMMON_TREE_DRAFT_SAFETENSORS_INDEX_MISSING_SHARD, "referenced Safetensors shard is missing");
        }
        const auto & shard = shard_it->second;
        const auto tensor_it = std::find_if(shard.table.tensors.begin(), shard.table.tensors.end(), [&](const auto & tensor) {
            return tensor.name == weight.first;
        });
        if (tensor_it == shard.table.tensors.end()) {
            return fail(COMMON_TREE_DRAFT_SAFETENSORS_INDEX_TENSOR_MISSING, "indexed tensor is missing from referenced shard");
        }
        result.directory.tensors.push_back({ *tensor_it, shard.source_path_index, shard.path });
    }

    return result;
}

const char * common_tree_draft_safetensors_index_error_name(common_tree_draft_safetensors_index_error error) {
    switch (error) {
        case COMMON_TREE_DRAFT_SAFETENSORS_INDEX_OK: return "ok";
        case COMMON_TREE_DRAFT_SAFETENSORS_INDEX_INVALID_SOURCE: return "invalid_source";
        case COMMON_TREE_DRAFT_SAFETENSORS_INDEX_MISSING_INDEX: return "missing_index";
        case COMMON_TREE_DRAFT_SAFETENSORS_INDEX_INVALID_JSON: return "invalid_json";
        case COMMON_TREE_DRAFT_SAFETENSORS_INDEX_INVALID_WEIGHT_MAP: return "invalid_weight_map";
        case COMMON_TREE_DRAFT_SAFETENSORS_INDEX_DUPLICATE_TENSOR: return "duplicate_tensor";
        case COMMON_TREE_DRAFT_SAFETENSORS_INDEX_MISSING_SHARD: return "missing_shard";
        case COMMON_TREE_DRAFT_SAFETENSORS_INDEX_SHARD_PARSE_ERROR: return "shard_parse_error";
        case COMMON_TREE_DRAFT_SAFETENSORS_INDEX_TENSOR_MISSING: return "tensor_missing";
        case COMMON_TREE_DRAFT_SAFETENSORS_INDEX_TENSOR_CONFLICT: return "tensor_conflict";
    }
    return "unknown";
}
