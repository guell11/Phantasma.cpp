#include "tree-draft-qkv.h"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

static void store_i32(std::vector<uint8_t> & storage, size_t offset, int32_t value) {
    std::memcpy(storage.data() + offset, &value, sizeof(value));
}

static int32_t load_i32(const std::vector<uint8_t> & storage, size_t index) {
    int32_t value = 0;
    std::memcpy(&value, storage.data() + index * sizeof(value), sizeof(value));
    return value;
}

static void fill_view(std::vector<uint8_t> & storage, const common_tree_draft_qkv_view & view, int32_t base) {
    for (int32_t g = 0; g < view.n_nodes; ++g) {
        for (int32_t h = 0; h < view.n_heads; ++h) {
            for (int32_t d = 0; d < view.head_dim; ++d) {
                const size_t offset = static_cast<size_t>(g) * view.stride_node +
                    static_cast<size_t>(h) * view.stride_head +
                    static_cast<size_t>(d) * view.stride_dim;
                store_i32(storage, offset, base + 100 * g + 10 * h + d);
            }
        }
    }
}

static void check_gather(
        const common_tree_draft_forest_offsets & forest,
        const int32_t * positions,
        const common_tree_draft_qkv_view & view,
        int32_t base) {
    bool overflow = true;
    const size_t bytes = common_tree_draft_qkv_packed_bytes(view, &overflow);
    assert(!overflow);
    std::vector<uint8_t> output(bytes + 8, 0xcd);
    assert(common_tree_draft_qkv_gather(forest, positions, static_cast<size_t>(view.n_nodes), view,
        { output.data(), output.size() }) == COMMON_TREE_DRAFT_QKV_OK);

    size_t index = 0;
    for (int32_t g = 0; g < view.n_nodes; ++g) {
        for (int32_t h = 0; h < view.n_heads; ++h) {
            for (int32_t d = 0; d < view.head_dim; ++d) {
                assert(load_i32(output, index++) == base + 100 * g + 10 * h + d);
            }
        }
    }
    for (size_t i = bytes; i < output.size(); ++i) {
        assert(output[i] == 0xcd);
    }
}

int main() {
    const int32_t parent[]  = { -1, 0, 0, -1, 3 };
    const int32_t depth[]   = {  0, 1, 1,  0, 1 };
    const int32_t tree_id[] = {  4, 4, 4,  9, 9 };
    const common_tree_draft_topology topology = { parent, depth, tree_id, 5 };
    const int32_t counts[] = { 3, 0, 2 };
    int32_t offsets[4] = {};
    assert(common_tree_draft_forest_offsets_build(topology, counts, 3, offsets, 4) == COMMON_TREE_DRAFT_FOREST_OK);
    const common_tree_draft_forest_offsets forest = { offsets, 3 };
    const int32_t prefixes[] = { 10, 88, 30 };
    int32_t positions[5] = {};
    assert(common_tree_draft_positions_build(topology, forest, prefixes, { positions, 5 }) == COMMON_TREE_DRAFT_POSITION_OK);

    std::vector<uint8_t> q_storage(5 * 96, 0xee);
    std::vector<uint8_t> k_storage(5 * 64, 0xee);
    std::vector<uint8_t> v_storage(5 * 80, 0xee);
    common_tree_draft_qkv_view q = { q_storage.data(), 5, 3, 2, 96, 24, 8, sizeof(int32_t) };
    common_tree_draft_qkv_view k = { k_storage.data(), 5, 2, 2, 64, 20, 8, sizeof(int32_t) };
    common_tree_draft_qkv_view v = { v_storage.data(), 5, 2, 3, 80, 28, 8, sizeof(int32_t) };
    fill_view(q_storage, q, 1000);
    fill_view(k_storage, k, 2000);
    fill_view(v_storage, v, 3000);

    check_gather(forest, positions, q, 1000);
    check_gather(forest, positions, k, 2000);
    check_gather(forest, positions, v, 3000);

    assert(common_tree_draft_qkv_gather(forest, nullptr, 5, q, { nullptr, 0 }) ==
        COMMON_TREE_DRAFT_QKV_NULL_POSITIONS);
    assert(common_tree_draft_qkv_gather(forest, positions, 4, q, { nullptr, 0 }) ==
        COMMON_TREE_DRAFT_QKV_POSITION_COUNT_MISMATCH);
    {
        auto bad = q;
        bad.n_nodes = 4;
        assert(common_tree_draft_qkv_gather(forest, positions, 5, bad, { nullptr, 0 }) ==
            COMMON_TREE_DRAFT_QKV_INVALID_SHAPE);
    }
    {
        auto bad = q;
        bad.data = nullptr;
        assert(common_tree_draft_qkv_gather(forest, positions, 5, bad, { nullptr, 0 }) ==
            COMMON_TREE_DRAFT_QKV_NULL_SOURCE);
    }
    {
        auto bad = q;
        bad.element_size = 0;
        assert(common_tree_draft_qkv_gather(forest, positions, 5, bad, { nullptr, 0 }) ==
            COMMON_TREE_DRAFT_QKV_ZERO_ELEMENT_SIZE);
    }
    {
        auto bad = q;
        bad.stride_node = std::numeric_limits<size_t>::max();
        assert(common_tree_draft_qkv_gather(forest, positions, 5, bad, { nullptr, 0 }) ==
            COMMON_TREE_DRAFT_QKV_STRIDE_OVERFLOW);
    }
    {
        bool overflow = false;
        const size_t required = common_tree_draft_qkv_packed_bytes(q, &overflow);
        assert(!overflow && required > 0);
        std::vector<uint8_t> short_output(required - 1, 0x5a);
        assert(common_tree_draft_qkv_gather(forest, positions, 5, q,
            { short_output.data(), short_output.size() }) == COMMON_TREE_DRAFT_QKV_OUTPUT_TOO_SMALL);
        for (uint8_t value : short_output) {
            assert(value == 0x5a);
        }
    }
    {
        const int32_t bad_offsets[] = { 0, 4, 3, 5 };
        assert(common_tree_draft_qkv_gather({ bad_offsets, 3 }, positions, 5, q, { nullptr, 0 }) ==
            COMMON_TREE_DRAFT_QKV_INVALID_FOREST);
    }
    {
        const int32_t empty_offsets[] = { 0 };
        const common_tree_draft_qkv_view empty = { nullptr, 0, 2, 4, 0, 0, 0, sizeof(float) };
        assert(common_tree_draft_qkv_gather({ empty_offsets, 0 }, nullptr, 0, empty, { nullptr, 0 }) ==
            COMMON_TREE_DRAFT_QKV_OK);
    }

    assert(std::string(common_tree_draft_qkv_error_name(COMMON_TREE_DRAFT_QKV_STRIDE_OVERFLOW)) == "stride_overflow");
    return 0;
}
