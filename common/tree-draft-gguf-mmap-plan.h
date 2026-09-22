#pragma once

#include "tree-draft-gguf-shards.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

enum common_tree_draft_gguf_mmap_plan_error : int32_t {
    COMMON_TREE_DRAFT_GGUF_MMAP_PLAN_OK = 0,
    COMMON_TREE_DRAFT_GGUF_MMAP_PLAN_INVALID_POLICY,
    COMMON_TREE_DRAFT_GGUF_MMAP_PLAN_SHARD_ERROR,
    COMMON_TREE_DRAFT_GGUF_MMAP_PLAN_TENSOR_ERROR,
    COMMON_TREE_DRAFT_GGUF_MMAP_PLAN_FILE_ERROR,
    COMMON_TREE_DRAFT_GGUF_MMAP_PLAN_OVERFLOW,
    COMMON_TREE_DRAFT_GGUF_MMAP_PLAN_OUT_OF_BOUNDS,
};

struct common_tree_draft_gguf_mmap_policy {
    uint64_t page_size = 4096;
    uint64_t max_gap = 0;
    uint64_t max_region_size = UINT64_MAX;
    bool coalesce = true;
};

struct common_tree_draft_gguf_mmap_tensor_binding {
    size_t logical_tensor_index = 0;
    std::string tensor_name;
    uint64_t file_offset = 0;
    uint64_t byte_size = 0;
    uint64_t intra_map_offset = 0;
};

struct common_tree_draft_gguf_mmap_region {
    uint16_t shard_index = 0;
    size_t source_path_index = 0;
    std::string source_path;
    uint64_t file_offset = 0;
    uint64_t length = 0;
    std::vector<common_tree_draft_gguf_mmap_tensor_binding> tensors;
};

struct common_tree_draft_gguf_mmap_plan {
    common_tree_draft_gguf_mmap_policy policy;
    std::vector<common_tree_draft_gguf_mmap_region> regions;
};

struct common_tree_draft_gguf_mmap_plan_result {
    common_tree_draft_gguf_mmap_plan plan;
    common_tree_draft_gguf_mmap_plan_error error = COMMON_TREE_DRAFT_GGUF_MMAP_PLAN_OK;
    size_t source_path_index = 0;
    std::string message;

    explicit operator bool() const {
        return error == COMMON_TREE_DRAFT_GGUF_MMAP_PLAN_OK;
    }
};

common_tree_draft_gguf_mmap_plan_result common_tree_draft_gguf_plan_mmap_regions(
        const common_tree_draft_model_source & source,
        const common_tree_draft_gguf_mmap_policy & policy = {});

const char * common_tree_draft_gguf_mmap_plan_error_name(common_tree_draft_gguf_mmap_plan_error error);
