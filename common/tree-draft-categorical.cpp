#include "tree-draft-categorical.h"

#include <cmath>
#include <limits>

common_tree_draft_categorical_status common_tree_draft_categorical_normalize(
        const float * logits,
        size_t n_vocab,
        float temperature,
        float * scaled_logits,
        float * probabilities) {
    if ((n_vocab > 0 && logits == nullptr) ||
        (n_vocab > 0 && scaled_logits == nullptr) ||
        (n_vocab > 0 && probabilities == nullptr)) {
        return COMMON_TREE_DRAFT_CATEGORICAL_NULL_BUFFER;
    }
    if (!(temperature > 0.0f) || !std::isfinite(temperature)) {
        return COMMON_TREE_DRAFT_CATEGORICAL_TEMPERATURE;
    }

    const float neg_inf = -std::numeric_limits<float>::infinity();
    double max_scaled = -std::numeric_limits<double>::infinity();
    size_t n_finite = 0;

    for (size_t i = 0; i < n_vocab; ++i) {
        const float logit = logits[i];
        if (std::isnan(logit) || logit == std::numeric_limits<float>::infinity()) {
            return COMMON_TREE_DRAFT_CATEGORICAL_INVALID_LOGIT;
        }
        if (logit == neg_inf) {
            scaled_logits[i] = neg_inf;
            probabilities[i] = 0.0f;
            continue;
        }

        const double scaled = static_cast<double>(logit) / static_cast<double>(temperature);
        if (!std::isfinite(scaled)) {
            return COMMON_TREE_DRAFT_CATEGORICAL_NUMERIC;
        }
        scaled_logits[i] = static_cast<float>(scaled);
        if (!std::isfinite(scaled_logits[i])) {
            return COMMON_TREE_DRAFT_CATEGORICAL_NUMERIC;
        }
        if (scaled > max_scaled) {
            max_scaled = scaled;
        }
        ++n_finite;
    }

    if (n_finite == 0) {
        return COMMON_TREE_DRAFT_CATEGORICAL_NO_FINITE;
    }

    double sum = 0.0;
    for (size_t i = 0; i < n_vocab; ++i) {
        if (scaled_logits[i] == neg_inf) {
            continue;
        }
        const double weight = std::exp(static_cast<double>(scaled_logits[i]) - max_scaled);
        if (!std::isfinite(weight)) {
            return COMMON_TREE_DRAFT_CATEGORICAL_NUMERIC;
        }
        probabilities[i] = static_cast<float>(weight);
        sum += weight;
    }
    if (!(sum > 0.0) || !std::isfinite(sum)) {
        return COMMON_TREE_DRAFT_CATEGORICAL_NUMERIC;
    }

    const double inv_sum = 1.0 / sum;
    for (size_t i = 0; i < n_vocab; ++i) {
        if (scaled_logits[i] != neg_inf) {
            probabilities[i] = static_cast<float>(static_cast<double>(probabilities[i]) * inv_sum);
        }
    }
    return COMMON_TREE_DRAFT_CATEGORICAL_OK;
}

