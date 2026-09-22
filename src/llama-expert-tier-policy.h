#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace llama_expert_tier {

struct hot_slot_policy_layer {
    size_t expert_bytes = 0;
    std::vector<float> scores;
    uint64_t misses = 0;
    uint64_t reuses = 0;
};

// Allocates a fixed byte budget over per-layer hot slots. Scores are ordered
// by expert rank; misses add demand not served by the current hot set and
// reuses discount demand already served from a lower tier. Empty signals
// return the uniform fallback unchanged.
std::vector<int> hot_slot_budget_policy(
        const std::vector<hot_slot_policy_layer> & layers,
        size_t byte_budget, int uniform_slots);

}
