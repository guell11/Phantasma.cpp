#pragma once

#include "tree-draft-model-source.h"
#include "tree-draft-storage-view.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

enum common_tree_draft_buffer_fallback_trigger : uint32_t {
    COMMON_TREE_DRAFT_BUFFER_TRIGGER_CAPABILITY = 0,
    COMMON_TREE_DRAFT_BUFFER_TRIGGER_POLICY,
    COMMON_TREE_DRAFT_BUFFER_TRIGGER_CORRUPT,
};

struct common_tree_draft_buffer_interval {
    size_t path_index = 0;
    uint64_t offset = 0;
    uint64_t length = 0;
    uint64_t alignment = 1;
};

struct common_tree_draft_buffered_payload {
    common_tree_draft_storage_view view;
    size_t path_index = 0;
    uint64_t source_offset = 0;
};

enum common_tree_draft_buffered_load_status : uint32_t {
    COMMON_TREE_DRAFT_BUFFERED_LOAD_OK = 0,
    COMMON_TREE_DRAFT_BUFFERED_LOAD_TRIGGER,
    COMMON_TREE_DRAFT_BUFFERED_LOAD_RANGE,
    COMMON_TREE_DRAFT_BUFFERED_LOAD_ALIGNMENT,
    COMMON_TREE_DRAFT_BUFFERED_LOAD_OPEN,
    COMMON_TREE_DRAFT_BUFFERED_LOAD_READ,
    COMMON_TREE_DRAFT_BUFFERED_LOAD_ALLOC,
};

common_tree_draft_buffered_load_status common_tree_draft_buffered_load(
        const common_tree_draft_model_source & source,
        const common_tree_draft_buffer_interval * intervals,
        size_t interval_count,
        common_tree_draft_buffer_fallback_trigger trigger,
        std::vector<common_tree_draft_buffered_payload> * payloads);

