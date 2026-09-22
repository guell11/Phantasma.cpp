#pragma once

#include "tree-draft-model-metadata.h"

#include <cstdint>
#include <optional>
#include <string>

enum class common_tree_draft_architecture_config_error {
    none,
    missing_architecture,
    unsupported_architecture,
    missing_n_layers,
    missing_n_embd,
    missing_n_heads,
    missing_n_heads_kv,
    missing_n_ff,
    missing_vocab_size,
    zero_n_layers,
    zero_n_embd,
    zero_n_heads,
    zero_n_heads_kv,
    zero_n_ff,
    zero_vocab_size,
    embedding_head_divisibility,
    grouped_query_divisibility,
    invalid_rope_theta,
};

class common_tree_draft_architecture_config {
public:
    common_tree_draft_architecture_config(const common_tree_draft_architecture_config &) = default;
    common_tree_draft_architecture_config(common_tree_draft_architecture_config &&) = default;

    common_tree_draft_architecture_config & operator=(const common_tree_draft_architecture_config &) = delete;
    common_tree_draft_architecture_config & operator=(common_tree_draft_architecture_config &&) = delete;

    static std::optional<common_tree_draft_architecture_config> normalize(
            const common_tree_draft_model_metadata & metadata,
            common_tree_draft_architecture_config_error * error = nullptr);

    const std::string & architecture() const;
    uint64_t n_layers() const;
    uint64_t n_embd() const;
    uint64_t n_heads() const;
    uint64_t n_heads_kv() const;
    uint64_t n_ff() const;
    uint64_t vocab_size() const;
    const std::optional<double> & rope_theta() const;

private:
    common_tree_draft_architecture_config(const common_tree_draft_model_metadata & metadata);

    const std::string architecture_;
    const uint64_t n_layers_;
    const uint64_t n_embd_;
    const uint64_t n_heads_;
    const uint64_t n_heads_kv_;
    const uint64_t n_ff_;
    const uint64_t vocab_size_;
    const std::optional<double> rope_theta_;
};

const char * common_tree_draft_architecture_config_error_name(common_tree_draft_architecture_config_error error);
