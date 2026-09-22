#pragma once

#include <cstdint>

// Stateless counter RNG address. The five words are also the tensor ABI order.
// Backends may consume rows of five I64 values in this exact order.
struct common_tree_draft_rng_address {
    uint64_t seed;
    uint64_t request;
    uint64_t step;
    uint64_t branch;
    uint64_t lane;
};

static constexpr uint32_t COMMON_TREE_DRAFT_RNG_ADDRESS_WORDS = 5;

static inline uint64_t common_tree_draft_rng_mix(uint64_t x) {
    x ^= x >> 30;
    x *= UINT64_C(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x *= UINT64_C(0x94d049bb133111eb);
    x ^= x >> 31;
    return x;
}

// Pure counter-based PRF. There is no mutable RNG state, so a logical draw is
// independent of batching, launch shape, stream ordering, and graph replay.
static inline uint64_t common_tree_draft_rng_u64(const common_tree_draft_rng_address & address) {
    uint64_t x = common_tree_draft_rng_mix(address.seed ^ UINT64_C(0x243f6a8885a308d3));
    x = common_tree_draft_rng_mix(x ^ common_tree_draft_rng_mix(address.request + UINT64_C(0x13198a2e03707344)));
    x = common_tree_draft_rng_mix(x ^ common_tree_draft_rng_mix(address.step    + UINT64_C(0xa4093822299f31d0)));
    x = common_tree_draft_rng_mix(x ^ common_tree_draft_rng_mix(address.branch  + UINT64_C(0x082efa98ec4e6c89)));
    x = common_tree_draft_rng_mix(x ^ common_tree_draft_rng_mix(address.lane    + UINT64_C(0x452821e638d01377)));
    return x;
}

// Use the high 24 bits so every result is exactly representable as float and
// lies in [0, 1). CPU and GPU implementations can reproduce this conversion.
static inline float common_tree_draft_rng_uniform_f32(const common_tree_draft_rng_address & address) {
    const uint32_t mantissa = (uint32_t) (common_tree_draft_rng_u64(address) >> 40);
    return (float) mantissa * (1.0f / 16777216.0f);
}
