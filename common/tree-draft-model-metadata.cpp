#include "tree-draft-model-metadata.h"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace {

common_tree_draft_model_metadata_origin origin(
        common_tree_draft_model_format format,
        const std::string & key) {
    return { format, key, 1.0f, false };
}

bool raw_u64(const common_tree_draft_gguf_metadata_value & value, uint64_t & result) {
    if (value.is_array || value.raw.empty() || value.raw.size() > sizeof(result)) {
        return false;
    }
    switch (value.type) {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_UINT64:
            result = 0;
            for (size_t i = 0; i < value.raw.size(); ++i) {
                result |= static_cast<uint64_t>(value.raw[i]) << (8 * i);
            }
            return true;
        default:
            return false;
    }
}

bool raw_f64(const common_tree_draft_gguf_metadata_value & value, double & result) {
    if (value.is_array) {
        return false;
    }
    if (value.type == GGUF_TYPE_FLOAT32 && value.raw.size() == sizeof(float)) {
        float parsed = 0.0f;
        std::memcpy(&parsed, value.raw.data(), sizeof(parsed));
        result = parsed;
        return true;
    }
    if (value.type == GGUF_TYPE_FLOAT64 && value.raw.size() == sizeof(double)) {
        std::memcpy(&result, value.raw.data(), sizeof(result));
        return true;
    }
    return false;
}

bool string_u64(const std::string & value, uint64_t & result) {
    if (value.empty() || value.front() == '-') {
        return false;
    }
    char * end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::strtoull(value.c_str(), &end, 10);
    if (errno == ERANGE || end == value.c_str() || *end != '\0') {
        return false;
    }
    if (parsed > std::numeric_limits<uint64_t>::max()) {
        return false;
    }
    result = static_cast<uint64_t>(parsed);
    return true;
}

bool string_f64(const std::string & value, double & result) {
    if (value.empty()) {
        return false;
    }
    char * end = nullptr;
    errno = 0;
    result = std::strtod(value.c_str(), &end);
    return errno != ERANGE && end != value.c_str() && *end == '\0';
}

void add_unknown(
        common_tree_draft_model_metadata & result,
        common_tree_draft_model_metadata_origin item_origin,
        const common_tree_draft_gguf_metadata_value & value) {
    result.unknown.push_back({ std::move(item_origin), value.type, value.raw, value.strings });
}

void add_unknown(
        common_tree_draft_model_metadata & result,
        common_tree_draft_model_metadata_origin item_origin,
        const std::string & value) {
    result.unknown.push_back({ std::move(item_origin), GGUF_TYPE_COUNT, {}, { value } });
}

}

common_tree_draft_model_metadata common_tree_draft_model_metadata_normalize(
        const common_tree_draft_gguf_metadata_table & table) {
    common_tree_draft_model_metadata result;
    std::string architecture;
    for (const auto & entry : table.entries) {
        if (entry.key == "general.architecture" && !entry.value.is_array && entry.value.type == GGUF_TYPE_STRING && entry.value.strings.size() == 1) {
            architecture = entry.value.strings[0];
            result.architecture = { architecture, origin(COMMON_TREE_DRAFT_MODEL_FORMAT_GGUF, entry.key) };
            continue;
        }
        if (entry.key == "general.quantization_version") {
            uint64_t value = 0;
            if (raw_u64(entry.value, value)) {
                result.quantization_version = { value, origin(COMMON_TREE_DRAFT_MODEL_FORMAT_GGUF, entry.key) };
                continue;
            }
        }
        add_unknown(result, origin(COMMON_TREE_DRAFT_MODEL_FORMAT_GGUF, entry.key), entry.value);
    }

    if (architecture.empty()) {
        return result;
    }

    result.unknown.clear();
    for (const auto & entry : table.entries) {
        const auto field = architecture + ".";
        uint64_t integer = 0;
        double floating = 0.0;
        if (entry.key == "general.architecture") {
            continue;
        }
        if (entry.key == "general.quantization_version") {
            if (result.quantization_version) {
                continue;
            }
            add_unknown(result, origin(COMMON_TREE_DRAFT_MODEL_FORMAT_GGUF, entry.key), entry.value);
            continue;
        }
        if (entry.key == field + "block_count" && raw_u64(entry.value, integer)) {
            result.n_layers = { integer, origin(COMMON_TREE_DRAFT_MODEL_FORMAT_GGUF, entry.key) };
        } else if (entry.key == field + "embedding_length" && raw_u64(entry.value, integer)) {
            result.n_embd = { integer, origin(COMMON_TREE_DRAFT_MODEL_FORMAT_GGUF, entry.key) };
        } else if (entry.key == field + "attention.head_count" && raw_u64(entry.value, integer)) {
            result.n_heads = { integer, origin(COMMON_TREE_DRAFT_MODEL_FORMAT_GGUF, entry.key) };
        } else if (entry.key == field + "attention.head_count_kv" && raw_u64(entry.value, integer)) {
            result.n_heads_kv = { integer, origin(COMMON_TREE_DRAFT_MODEL_FORMAT_GGUF, entry.key) };
        } else if (entry.key == field + "feed_forward_length" && raw_u64(entry.value, integer)) {
            result.n_ff = { integer, origin(COMMON_TREE_DRAFT_MODEL_FORMAT_GGUF, entry.key) };
        } else if (entry.key == field + "vocab_size" && raw_u64(entry.value, integer)) {
            result.vocab_size = { integer, origin(COMMON_TREE_DRAFT_MODEL_FORMAT_GGUF, entry.key) };
        } else if (entry.key == field + "rope.freq_base" && raw_f64(entry.value, floating)) {
            result.rope_theta = { floating, origin(COMMON_TREE_DRAFT_MODEL_FORMAT_GGUF, entry.key) };
        } else {
            add_unknown(result, origin(COMMON_TREE_DRAFT_MODEL_FORMAT_GGUF, entry.key), entry.value);
        }
    }
    return result;
}

common_tree_draft_model_metadata common_tree_draft_model_metadata_normalize(
        const common_tree_draft_safetensors_table & table) {
    common_tree_draft_model_metadata result;
    for (const auto & item : table.metadata) {
        const auto item_origin = origin(COMMON_TREE_DRAFT_MODEL_FORMAT_SAFETENSORS, item.first);
        uint64_t integer = 0;
        double floating = 0.0;
        if ((item.first == "architecture" || item.first == "model_type") && !result.architecture) {
            result.architecture = { item.second, item_origin };
        } else if (item.first == "num_hidden_layers" && string_u64(item.second, integer)) {
            result.n_layers = { integer, item_origin };
        } else if (item.first == "hidden_size" && string_u64(item.second, integer)) {
            result.n_embd = { integer, item_origin };
        } else if (item.first == "num_attention_heads" && string_u64(item.second, integer)) {
            result.n_heads = { integer, item_origin };
        } else if (item.first == "num_key_value_heads" && string_u64(item.second, integer)) {
            result.n_heads_kv = { integer, item_origin };
        } else if (item.first == "intermediate_size" && string_u64(item.second, integer)) {
            result.n_ff = { integer, item_origin };
        } else if (item.first == "vocab_size" && string_u64(item.second, integer)) {
            result.vocab_size = { integer, item_origin };
        } else if (item.first == "rope_theta" && string_f64(item.second, floating)) {
            result.rope_theta = { floating, item_origin };
        } else if (item.first == "quantization_version" && string_u64(item.second, integer)) {
            result.quantization_version = { integer, item_origin };
        } else {
            add_unknown(result, item_origin, item.second);
        }
    }
    return result;
}
