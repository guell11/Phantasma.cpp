#include "tree-draft-architecture-config.h"

#include <cmath>

namespace {

bool supported_architecture(const std::string & architecture) {
    return architecture == "llama" || architecture == "mistral" ||
           architecture == "gemma" || architecture == "gemma2" ||
           architecture == "gemma3" || architecture == "gemma4" ||
           architecture == "qwen2" || architecture == "qwen3" ||
           architecture == "phi3";
}

}

std::optional<common_tree_draft_architecture_config> common_tree_draft_architecture_config::normalize(
        const common_tree_draft_model_metadata & metadata,
        common_tree_draft_architecture_config_error * error) {
    const auto reject = [error](common_tree_draft_architecture_config_error value) {
        if (error != nullptr) {
            *error = value;
        }
        return std::optional<common_tree_draft_architecture_config>{};
    };

    if (!metadata.architecture || metadata.architecture->value.empty()) {
        return reject(common_tree_draft_architecture_config_error::missing_architecture);
    }
    if (!supported_architecture(metadata.architecture->value)) {
        return reject(common_tree_draft_architecture_config_error::unsupported_architecture);
    }
    if (!metadata.n_layers) {
        return reject(common_tree_draft_architecture_config_error::missing_n_layers);
    }
    if (!metadata.n_embd) {
        return reject(common_tree_draft_architecture_config_error::missing_n_embd);
    }
    if (!metadata.n_heads) {
        return reject(common_tree_draft_architecture_config_error::missing_n_heads);
    }
    if (!metadata.n_heads_kv) {
        return reject(common_tree_draft_architecture_config_error::missing_n_heads_kv);
    }
    if (!metadata.n_ff) {
        return reject(common_tree_draft_architecture_config_error::missing_n_ff);
    }
    if (!metadata.vocab_size) {
        return reject(common_tree_draft_architecture_config_error::missing_vocab_size);
    }
    if (metadata.n_layers->value == 0) {
        return reject(common_tree_draft_architecture_config_error::zero_n_layers);
    }
    if (metadata.n_embd->value == 0) {
        return reject(common_tree_draft_architecture_config_error::zero_n_embd);
    }
    if (metadata.n_heads->value == 0) {
        return reject(common_tree_draft_architecture_config_error::zero_n_heads);
    }
    if (metadata.n_heads_kv->value == 0) {
        return reject(common_tree_draft_architecture_config_error::zero_n_heads_kv);
    }
    if (metadata.n_ff->value == 0) {
        return reject(common_tree_draft_architecture_config_error::zero_n_ff);
    }
    if (metadata.vocab_size->value == 0) {
        return reject(common_tree_draft_architecture_config_error::zero_vocab_size);
    }
    if (metadata.n_embd->value % metadata.n_heads->value != 0) {
        return reject(common_tree_draft_architecture_config_error::embedding_head_divisibility);
    }
    if (metadata.n_heads->value % metadata.n_heads_kv->value != 0) {
        return reject(common_tree_draft_architecture_config_error::grouped_query_divisibility);
    }
    if (metadata.rope_theta && (!(metadata.rope_theta->value > 0.0) || !std::isfinite(metadata.rope_theta->value))) {
        return reject(common_tree_draft_architecture_config_error::invalid_rope_theta);
    }

    if (error != nullptr) {
        *error = common_tree_draft_architecture_config_error::none;
    }
    return common_tree_draft_architecture_config(metadata);
}

common_tree_draft_architecture_config::common_tree_draft_architecture_config(
        const common_tree_draft_model_metadata & metadata)
    : architecture_(metadata.architecture->value),
      n_layers_(metadata.n_layers->value),
      n_embd_(metadata.n_embd->value),
      n_heads_(metadata.n_heads->value),
      n_heads_kv_(metadata.n_heads_kv->value),
      n_ff_(metadata.n_ff->value),
      vocab_size_(metadata.vocab_size->value),
      rope_theta_(metadata.rope_theta ? std::optional<double>(metadata.rope_theta->value) : std::nullopt) {
}

const std::string & common_tree_draft_architecture_config::architecture() const { return architecture_; }
uint64_t common_tree_draft_architecture_config::n_layers() const { return n_layers_; }
uint64_t common_tree_draft_architecture_config::n_embd() const { return n_embd_; }
uint64_t common_tree_draft_architecture_config::n_heads() const { return n_heads_; }
uint64_t common_tree_draft_architecture_config::n_heads_kv() const { return n_heads_kv_; }
uint64_t common_tree_draft_architecture_config::n_ff() const { return n_ff_; }
uint64_t common_tree_draft_architecture_config::vocab_size() const { return vocab_size_; }
const std::optional<double> & common_tree_draft_architecture_config::rope_theta() const { return rope_theta_; }

const char * common_tree_draft_architecture_config_error_name(common_tree_draft_architecture_config_error error) {
    switch (error) {
        case common_tree_draft_architecture_config_error::none:                         return "none";
        case common_tree_draft_architecture_config_error::missing_architecture:         return "missing_architecture";
        case common_tree_draft_architecture_config_error::unsupported_architecture:     return "unsupported_architecture";
        case common_tree_draft_architecture_config_error::missing_n_layers:             return "missing_n_layers";
        case common_tree_draft_architecture_config_error::missing_n_embd:               return "missing_n_embd";
        case common_tree_draft_architecture_config_error::missing_n_heads:              return "missing_n_heads";
        case common_tree_draft_architecture_config_error::missing_n_heads_kv:           return "missing_n_heads_kv";
        case common_tree_draft_architecture_config_error::missing_n_ff:                 return "missing_n_ff";
        case common_tree_draft_architecture_config_error::missing_vocab_size:           return "missing_vocab_size";
        case common_tree_draft_architecture_config_error::zero_n_layers:                return "zero_n_layers";
        case common_tree_draft_architecture_config_error::zero_n_embd:                  return "zero_n_embd";
        case common_tree_draft_architecture_config_error::zero_n_heads:                 return "zero_n_heads";
        case common_tree_draft_architecture_config_error::zero_n_heads_kv:              return "zero_n_heads_kv";
        case common_tree_draft_architecture_config_error::zero_n_ff:                    return "zero_n_ff";
        case common_tree_draft_architecture_config_error::zero_vocab_size:              return "zero_vocab_size";
        case common_tree_draft_architecture_config_error::embedding_head_divisibility:  return "embedding_head_divisibility";
        case common_tree_draft_architecture_config_error::grouped_query_divisibility:   return "grouped_query_divisibility";
        case common_tree_draft_architecture_config_error::invalid_rope_theta:           return "invalid_rope_theta";
    }
    return "unknown";
}
