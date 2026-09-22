#include "../src/llama-expert-tier-policy.h"

#include <cstdio>
#include <vector>

static bool same(const std::vector<int> & a, const std::vector<int> & b) {
    return a == b;
}

int main() {
    using llama_expert_tier::hot_slot_budget_policy;
    using llama_expert_tier::hot_slot_policy_layer;

    std::vector<hot_slot_policy_layer> fallback(2);
    fallback[0].expert_bytes = 10;
    fallback[0].scores = { 0.0f, 0.0f };
    fallback[1].expert_bytes = 20;
    fallback[1].scores = { 0.0f, 0.0f };
    if (!same(hot_slot_budget_policy(fallback, 60, 2), { 2, 2 })) {
        std::fputs("uniform fallback mismatch\n", stderr);
        return 1;
    }

    std::vector<hot_slot_policy_layer> scored = fallback;
    scored[0].scores = { 9.0f, 1.0f };
    scored[1].scores = { 20.0f, 2.0f };
    if (!same(hot_slot_budget_policy(scored, 40, 2), { 2, 1 })) {
        std::fputs("score/byte allocation mismatch\n", stderr);
        return 1;
    }

    std::vector<hot_slot_policy_layer> miss(2);
    miss[0].expert_bytes = 10;
    miss[0].scores = { 0.0f, 0.0f };
    miss[0].misses = 8;
    miss[0].reuses = 2;
    miss[1].expert_bytes = 10;
    miss[1].scores = { 0.0f, 0.0f };
    miss[1].misses = 2;
    miss[1].reuses = 2;
    if (!same(hot_slot_budget_policy(miss, 20, 1), { 2, 0 })) {
        std::fputs("miss/reuse allocation mismatch\n", stderr);
        return 1;
    }

    std::puts("wackMall tier policy: OK");
    return 0;
}
