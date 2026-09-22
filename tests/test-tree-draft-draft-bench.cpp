#include "tree-draft-multi-sample.h"
#include "tree-draft-path.h"
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

struct branch_state {
    uint64_t path = 0;
    double logp = 0.0;
};

static std::vector<common_tree_draft_candidate> make_candidates(size_t n) {
    std::vector<common_tree_draft_candidate> out(n);
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double p = std::exp(-0.35 * static_cast<double>(i));
        out[i] = {static_cast<int32_t>(1000 + i), static_cast<float>(std::log(p)), static_cast<float>(p)};
        sum += p;
    }
    for (auto & c : out) c.probability = static_cast<float>(c.probability / sum);
    return out;
}

static double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

static void bench_linear(const std::vector<common_tree_draft_candidate> & candidates, int iterations, int depth) {
    uint64_t sink = 0;
    uint64_t proposals = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < iterations; ++it) {
        uint64_t path = common_tree_draft_path_root(1234 + static_cast<uint64_t>(it));
        for (int d = 1; d <= depth; ++d) {
            common_tree_draft_child_sample s{};
            size_t n = 0;
            if (common_tree_draft_sample_without_replacement(candidates.data(), candidates.size(), 1,
                    0x12345678ULL, static_cast<uint64_t>(it), path, static_cast<uint32_t>(d), &s, 1, &n) != COMMON_TREE_DRAFT_MULTI_SAMPLE_OK || n != 1) {
                std::abort();
            }
            uint64_t child = 0;
            if (common_tree_draft_path_child(path, s.token, 0, static_cast<uint32_t>(d), &child) != COMMON_TREE_DRAFT_PATH_OK) std::abort();
            path = child;
            sink ^= path;
            ++proposals;
        }
    }
    const double sec = seconds_since(t0);
    std::printf("linear_mtp iterations=%d depth=%d proposals=%llu seconds=%.6f proposals_per_s=%.3f ns_per_proposal=%.3f sink=%llu\n",
        iterations, depth, (unsigned long long) proposals, sec, proposals / sec, sec * 1e9 / proposals, (unsigned long long) sink);
}

static void bench_tree(const std::vector<common_tree_draft_candidate> & candidates, int iterations, int depth, int beam, int width) {
    uint64_t sink = 0;
    uint64_t proposals = 0;
    std::vector<branch_state> frontier;
    std::vector<branch_state> next;
    frontier.reserve(static_cast<size_t>(beam));
    next.reserve(static_cast<size_t>(beam) * static_cast<size_t>(width));
    std::vector<common_tree_draft_child_sample> samples(static_cast<size_t>(width));

    const auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < iterations; ++it) {
        frontier.clear();
        frontier.push_back({common_tree_draft_path_root(1234 + static_cast<uint64_t>(it)), 0.0});
        for (int d = 1; d <= depth; ++d) {
            next.clear();
            for (const auto & parent : frontier) {
                size_t n = 0;
                if (common_tree_draft_sample_without_replacement(candidates.data(), candidates.size(), static_cast<size_t>(width),
                        0x12345678ULL, static_cast<uint64_t>(it), parent.path, static_cast<uint32_t>(d), samples.data(), samples.size(), &n) != COMMON_TREE_DRAFT_MULTI_SAMPLE_OK) {
                    std::abort();
                }
                for (size_t j = 0; j < n; ++j) {
                    uint64_t child = 0;
                    if (common_tree_draft_path_child(parent.path, samples[j].token, samples[j].draw_ordinal,
                            static_cast<uint32_t>(d), &child) != COMMON_TREE_DRAFT_PATH_OK) std::abort();
                    next.push_back({child, parent.logp + samples[j].log_conditional_probability});
                    ++proposals;
                }
            }
            std::sort(next.begin(), next.end(), [](const branch_state & a, const branch_state & b) {
                if (a.logp != b.logp) return a.logp > b.logp;
                return a.path < b.path;
            });
            if (next.size() > static_cast<size_t>(beam)) next.resize(static_cast<size_t>(beam));
            frontier.swap(next);
        }
        for (const auto & b : frontier) sink ^= b.path;
    }
    const double sec = seconds_since(t0);
    std::printf("tree_draft iterations=%d depth=%d beam=%d width=%d proposals=%llu seconds=%.6f proposals_per_s=%.3f ns_per_proposal=%.3f sink=%llu\n",
        iterations, depth, beam, width, (unsigned long long) proposals, sec, proposals / sec, sec * 1e9 / proposals, (unsigned long long) sink);
}

static void bench_model_load(const std::string & model_path, bool gpu) {
    llama_backend_init();
    auto p = llama_model_default_params();
    p.n_gpu_layers = gpu ? 999 : 0;
    const auto t0 = std::chrono::steady_clock::now();
    llama_model * model = llama_model_load_from_file(model_path.c_str(), p);
    const double sec = seconds_since(t0);
    std::printf("draft_model_load path=%s device=%s seconds=%.6f ok=%d\n", model_path.c_str(), gpu ? "gpu" : "cpu", sec, model != nullptr ? 1 : 0);
    if (model) llama_model_free(model);
    llama_backend_free();
}

int main(int argc, char ** argv) {
    int iterations = 200000;
    int depth = 4;
    int beam = 4;
    int width = 4;
    int candidates_n = 16;
    std::string model;
    bool gpu = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next_i = [&](int & dst) { if (i + 1 < argc) dst = std::atoi(argv[++i]); };
        if (a == "--iterations") next_i(iterations);
        else if (a == "--depth") next_i(depth);
        else if (a == "--beam") next_i(beam);
        else if (a == "--width") next_i(width);
        else if (a == "--candidates") next_i(candidates_n);
        else if (a == "--model" && i + 1 < argc) model = argv[++i];
        else if (a == "--gpu") gpu = true;
    }
    if (iterations <= 0 || depth <= 0 || beam <= 0 || width <= 0 || candidates_n < width) return 2;

    auto candidates = make_candidates(static_cast<size_t>(candidates_n));
    bench_linear(candidates, iterations, depth);
    bench_tree(candidates, iterations, depth, beam, width);
    if (!model.empty()) bench_model_load(model, gpu);
    return 0;
}
