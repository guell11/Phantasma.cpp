#include "tree-draft-ragged-attention.h"

#include <cassert>
#include <cstdint>
#include <limits>
#include <string>

int main() {
    const int32_t offsets[] = { 0, 3, 3, 7 };
    const common_tree_draft_forest_offsets forest = { offsets, 3 };
    const int32_t prefix_lengths[] = { 11, 0, 29 };
    const int32_t kv_bases[] = { 100, 500, 900 };
    common_tree_draft_ragged_attention_entry entries[3] = {};
    common_tree_draft_ragged_attention_batch batch = { entries, 3, -1, -1, -1 };

    assert(common_tree_draft_ragged_attention_build(forest, prefix_lengths, kv_bases, batch) ==
        COMMON_TREE_DRAFT_RAGGED_ATTENTION_OK);
    assert(batch.n_entries == 3);
    assert(batch.n_queries == 7);
    assert(batch.n_tree_nodes == 7);
    assert(common_tree_draft_ragged_attention_validate(batch) == COMMON_TREE_DRAFT_RAGGED_ATTENTION_OK);

    assert(entries[0].q_offset == 0 && entries[0].q_len == 3);
    assert(entries[0].prefix_len == 11 && entries[0].tree_offset == 0 && entries[0].tree_len == 3);
    assert(entries[0].kv_base == 100);
    assert(entries[1].q_offset == 3 && entries[1].q_len == 0);
    assert(entries[1].prefix_len == 0 && entries[1].tree_offset == 3 && entries[1].tree_len == 0);
    assert(entries[1].kv_base == 500);
    assert(entries[2].q_offset == 3 && entries[2].q_len == 4);
    assert(entries[2].prefix_len == 29 && entries[2].tree_offset == 3 && entries[2].tree_len == 4);
    assert(entries[2].kv_base == 900);

    // Caller-owned storage and fixed POD fields make repeated graph-capture preparation stable.
    const auto * stable_entries = batch.entries;
    assert(common_tree_draft_ragged_attention_build(forest, prefix_lengths, kv_bases, batch) ==
        COMMON_TREE_DRAFT_RAGGED_ATTENTION_OK);
    assert(batch.entries == stable_entries);

    {
        const int32_t empty_offsets[] = { 0, 0 };
        const common_tree_draft_forest_offsets empty_forest = { empty_offsets, 1 };
        const int32_t empty_prefix[] = { 8 };
        const int32_t empty_kv[] = { 17 };
        common_tree_draft_ragged_attention_entry empty_entry = {};
        common_tree_draft_ragged_attention_batch empty_batch = { &empty_entry, 1, -1, -1, -1 };
        assert(common_tree_draft_ragged_attention_build(empty_forest, empty_prefix, empty_kv, empty_batch) ==
            COMMON_TREE_DRAFT_RAGGED_ATTENTION_OK);
        assert(empty_batch.n_queries == 0 && empty_batch.n_tree_nodes == 0);
        assert(empty_entry.q_offset == 0 && empty_entry.q_len == 0 && empty_entry.tree_len == 0);
        assert(empty_entry.prefix_len == 8 && empty_entry.kv_base == 17);
    }

    {
        const int32_t bad_offsets[] = { 0, 4, 3 };
        assert(common_tree_draft_ragged_attention_build({ bad_offsets, 2 }, prefix_lengths, kv_bases, batch) ==
            COMMON_TREE_DRAFT_RAGGED_ATTENTION_INVALID_FOREST);
    }
    assert(common_tree_draft_ragged_attention_build(forest, nullptr, kv_bases, batch) ==
        COMMON_TREE_DRAFT_RAGGED_ATTENTION_NULL_PREFIX_LENGTHS);
    assert(common_tree_draft_ragged_attention_build(forest, prefix_lengths, nullptr, batch) ==
        COMMON_TREE_DRAFT_RAGGED_ATTENTION_NULL_KV_BASES);
    {
        const int32_t bad_prefix[] = { 11, -1, 29 };
        assert(common_tree_draft_ragged_attention_build(forest, bad_prefix, kv_bases, batch) ==
            COMMON_TREE_DRAFT_RAGGED_ATTENTION_NEGATIVE_PREFIX_LENGTH);
    }
    {
        const int32_t bad_kv[] = { 100, -1, 900 };
        assert(common_tree_draft_ragged_attention_build(forest, prefix_lengths, bad_kv, batch) ==
            COMMON_TREE_DRAFT_RAGGED_ATTENTION_NEGATIVE_KV_BASE);
    }
    {
        common_tree_draft_ragged_attention_batch no_output = { nullptr, 3, 0, 0, 0 };
        assert(common_tree_draft_ragged_attention_build(forest, prefix_lengths, kv_bases, no_output) ==
            COMMON_TREE_DRAFT_RAGGED_ATTENTION_NULL_OUTPUT);
        common_tree_draft_ragged_attention_batch short_output = { entries, 2, 0, 0, 0 };
        assert(common_tree_draft_ragged_attention_build(forest, prefix_lengths, kv_bases, short_output) ==
            COMMON_TREE_DRAFT_RAGGED_ATTENTION_OUTPUT_TOO_SMALL);
    }
    {
        const int32_t overflow_prefix[] = { 11, 0, std::numeric_limits<int32_t>::max() };
        assert(common_tree_draft_ragged_attention_build(forest, overflow_prefix, kv_bases, batch) ==
            COMMON_TREE_DRAFT_RAGGED_ATTENTION_RANGE_OVERFLOW);
    }
    {
        const int32_t overflow_kv[] = { 100, 500, std::numeric_limits<int32_t>::max() - 30 };
        assert(common_tree_draft_ragged_attention_build(forest, prefix_lengths, overflow_kv, batch) ==
            COMMON_TREE_DRAFT_RAGGED_ATTENTION_RANGE_OVERFLOW);
    }
    {
        const auto saved = entries[2];
        entries[2].q_offset = 4;
        assert(common_tree_draft_ragged_attention_validate(batch) == COMMON_TREE_DRAFT_RAGGED_ATTENTION_INVALID_FOREST);
        entries[2] = saved;
        assert(common_tree_draft_ragged_attention_validate(batch) == COMMON_TREE_DRAFT_RAGGED_ATTENTION_OK);
    }

    assert(std::string(common_tree_draft_ragged_attention_error_name(
        COMMON_TREE_DRAFT_RAGGED_ATTENTION_RANGE_OVERFLOW)) == "range_overflow");
    return 0;
}
