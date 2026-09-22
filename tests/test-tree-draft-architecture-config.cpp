#include "tree-draft-architecture-config.h"

#include <cassert>
#include <limits>
#include <type_traits>

static common_tree_draft_model_metadata_u64 u64(uint64_t value) {
    return { value, {} };
}

static common_tree_draft_model_metadata valid_metadata() {
    common_tree_draft_model_metadata metadata;
    metadata.architecture = { "gemma4", {} };
    metadata.n_layers = u64(34);
    metadata.n_embd = u64(4096);
    metadata.n_heads = u64(32);
    metadata.n_heads_kv = u64(8);
    metadata.n_ff = u64(16384);
    metadata.vocab_size = u64(256000);
    metadata.rope_theta = { 1000000.0, {} };
    return metadata;
}

static void test_valid_config_is_immutable() {
    static_assert(!std::is_default_constructible<common_tree_draft_architecture_config>::value, "config requires normalization");
    static_assert(!std::is_copy_assignable<common_tree_draft_architecture_config>::value, "config must be immutable");

    common_tree_draft_architecture_config_error error = common_tree_draft_architecture_config_error::missing_architecture;
    const auto config = common_tree_draft_architecture_config::normalize(valid_metadata(), &error);
    assert(config);
    assert(error == common_tree_draft_architecture_config_error::none);
    assert(config->architecture() == "gemma4");
    assert(config->n_layers() == 34);
    assert(config->n_embd() == 4096);
    assert(config->n_heads() == 32);
    assert(config->n_heads_kv() == 8);
    assert(config->n_ff() == 16384);
    assert(config->vocab_size() == 256000);
    assert(config->rope_theta() && *config->rope_theta() == 1000000.0);
}

static void expect_error(
        common_tree_draft_model_metadata metadata,
        common_tree_draft_architecture_config_error expected) {
    common_tree_draft_architecture_config_error error = common_tree_draft_architecture_config_error::none;
    assert(!common_tree_draft_architecture_config::normalize(metadata, &error));
    assert(error == expected);
}

static void test_missing_and_invalid_dimensions() {
    auto metadata = valid_metadata();
    metadata.n_heads_kv.reset();
    expect_error(metadata, common_tree_draft_architecture_config_error::missing_n_heads_kv);

    metadata = valid_metadata();
    metadata.n_layers = u64(0);
    expect_error(metadata, common_tree_draft_architecture_config_error::zero_n_layers);

    metadata = valid_metadata();
    metadata.n_embd = u64(4097);
    expect_error(metadata, common_tree_draft_architecture_config_error::embedding_head_divisibility);

    metadata = valid_metadata();
    metadata.n_heads_kv = u64(6);
    expect_error(metadata, common_tree_draft_architecture_config_error::grouped_query_divisibility);

    metadata = valid_metadata();
    metadata.rope_theta = { std::numeric_limits<double>::infinity(), {} };
    expect_error(metadata, common_tree_draft_architecture_config_error::invalid_rope_theta);
}

static void test_architecture_is_explicit() {
    auto metadata = valid_metadata();
    metadata.architecture.reset();
    expect_error(metadata, common_tree_draft_architecture_config_error::missing_architecture);

    metadata = valid_metadata();
    metadata.architecture = { "unknown", {} };
    expect_error(metadata, common_tree_draft_architecture_config_error::unsupported_architecture);
}

int main() {
    test_valid_config_is_immutable();
    test_missing_and_invalid_dimensions();
    test_architecture_is_explicit();
    return 0;
}
