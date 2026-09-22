#include "../src/llama-expert-tier.h"

#include <cmath>
#include <cstdio>

static bool equal(float a, float b) {
    return std::fabs(a - b) <= 1e-6f;
}

int main() {
    const float first = llama_expert_tier::score_ema_update(4.0f, 0.5f, 2);
    const float pp_idle = llama_expert_tier::score_ema_update(first, 0.5f, 0);
    const float tg_hit = llama_expert_tier::score_ema_update(first, 0.5f, 3);

    if (!equal(first, 4.0f) || !equal(pp_idle, 2.0f) || !equal(tg_hit, 5.0f)) {
        std::fprintf(stderr, "wackMall tier score mismatch: first=%g idle=%g hit=%g\n", first, pp_idle, tg_hit);
        return 1;
    }

    std::puts("wackMall tier score: OK");
    return 0;
}
