#pragma once

#include "tree-draft-gguf-metadata.h"
#include "tree-draft-safetensors.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

struct common_tree_draft_model_metadata_origin {
    common_tree_draft_model_format format = COMMON_TREE_DRAFT_MODEL_FORMAT_UNKNOWN;
    std::string key;
    float confidence = 1.0f;
    bool inferred = false;
};

struct common_tree_draft_model_metadata_text {
    std::string value;
    common_tree_draft_model_metadata_origin origin;
};

struct common_tree_draft_model_metadata_u64 {
    uint64_t value = 0;
    common_tree_draft_model_metadata_origin origin;
};

struct common_tree_draft_model_metadata_f64 {
    double value = 0.0;
    common_tree_draft_model_metadata_origin origin;
};

struct common_tree_draft_model_metadata_unknown {
    common_tree_draft_model_metadata_origin origin;
    gguf_type gguf_type = GGUF_TYPE_COUNT;
    std::vector<unsigned char> raw;
    std::vector<std::string> strings;
};

struct common_tree_draft_model_metadata {
    std::optional<common_tree_draft_model_metadata_text> architecture;
    std::optional<common_tree_draft_model_metadata_u64> n_layers;
    std::optional<common_tree_draft_model_metadata_u64> n_embd;
    std::optional<common_tree_draft_model_metadata_u64> n_heads;
    std::optional<common_tree_draft_model_metadata_u64> n_heads_kv;
    std::optional<common_tree_draft_model_metadata_u64> n_ff;
    std::optional<common_tree_draft_model_metadata_u64> vocab_size;
    std::optional<common_tree_draft_model_metadata_f64> rope_theta;
    std::optional<common_tree_draft_model_metadata_u64> quantization_version;
    std::vector<common_tree_draft_model_metadata_unknown> unknown;
};

common_tree_draft_model_metadata common_tree_draft_model_metadata_normalize(
        const common_tree_draft_gguf_metadata_table & table);

common_tree_draft_model_metadata common_tree_draft_model_metadata_normalize(
        const common_tree_draft_safetensors_table & table);
