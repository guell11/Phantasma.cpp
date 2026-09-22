#pragma once

#include "tree-draft-quant-registry.h"

#include <cstdint>
#include <vector>

enum common_tree_draft_dequant_accumulation_mode : uint32_t {
    COMMON_TREE_DRAFT_DEQUANT_ACCUM_DETERMINISTIC = 0,
    COMMON_TREE_DRAFT_DEQUANT_ACCUM_THROUGHPUT,
};

struct common_tree_draft_dequant_tile {
    uint64_t logical_begin = 0;
    uint64_t logical_count = 0;
    uint64_t encoded_offset = 0;
    uint64_t encoded_bytes = 0;
    uint64_t temporary_bytes = 0;
};

struct common_tree_draft_dequant_plan {
    std::vector<common_tree_draft_dequant_tile> tiles;
    uint64_t peak_temporary_bytes = 0;
    common_tree_draft_dequant_accumulation_mode mode = COMMON_TREE_DRAFT_DEQUANT_ACCUM_DETERMINISTIC;
};

enum common_tree_draft_dequant_plan_status : uint32_t {
    COMMON_TREE_DRAFT_DEQUANT_PLAN_OK = 0,
    COMMON_TREE_DRAFT_DEQUANT_PLAN_INVALID,
    COMMON_TREE_DRAFT_DEQUANT_PLAN_BUDGET,
    COMMON_TREE_DRAFT_DEQUANT_PLAN_OVERFLOW,
};

common_tree_draft_dequant_plan_status common_tree_draft_dequant_plan_build(
        const common_tree_draft_quant_type & quant,
        uint64_t logical_elements,
        uint64_t workspace_budget,
        uint32_t output_element_bytes,
        common_tree_draft_dequant_accumulation_mode mode,
        common_tree_draft_dequant_plan * plan);

