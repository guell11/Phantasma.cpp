#include "tree-draft-kv-window.h"

#include <algorithm>
#include <limits>

static common_tree_draft_kv_window_status common_tree_draft_kv_window_validate(
        const common_tree_draft_kv_gather_metadata & source) {
    if (source.row_ptr == nullptr || (source.segment_count > 0 && source.segments == nullptr)) {
        return COMMON_TREE_DRAFT_KV_WINDOW_METADATA;
    }
    if (source.row_ptr_capacity < static_cast<size_t>(source.query_count) + 1 ||
        source.segment_capacity < source.segment_count || source.row_ptr[0] != 0 ||
        source.row_ptr[source.query_count] != source.segment_count) {
        return COMMON_TREE_DRAFT_KV_WINDOW_METADATA;
    }
    for (uint32_t q = 0; q < source.query_count; ++q) {
        if (source.row_ptr[q] > source.row_ptr[q + 1] || source.row_ptr[q + 1] > source.segment_count) {
            return COMMON_TREE_DRAFT_KV_WINDOW_METADATA;
        }
    }
    for (uint32_t i = 0; i < source.segment_count; ++i) {
        const auto & segment = source.segments[i];
        if (segment.page.id == COMMON_TREE_DRAFT_KV_ID_INVALID || segment.lo >= segment.hi) {
            return COMMON_TREE_DRAFT_KV_WINDOW_METADATA;
        }
    }
    return COMMON_TREE_DRAFT_KV_WINDOW_OK;
}

common_tree_draft_kv_window_status common_tree_draft_kv_window_clip(
        const common_tree_draft_kv_gather_metadata & source,
        const common_tree_draft_kv_window_query * queries,
        size_t query_count,
        common_tree_draft_kv_gather_metadata * output) {
    if (output == nullptr || (query_count > 0 && queries == nullptr)) {
        return COMMON_TREE_DRAFT_KV_WINDOW_NULL_BUFFER;
    }
    const auto source_status = common_tree_draft_kv_window_validate(source);
    if (source_status != COMMON_TREE_DRAFT_KV_WINDOW_OK) return source_status;
    if (query_count != source.query_count) return COMMON_TREE_DRAFT_KV_WINDOW_QUERY_RANGE;
    if (output->row_ptr == nullptr || output->row_ptr_capacity < query_count + 1) {
        return COMMON_TREE_DRAFT_KV_WINDOW_OUTPUT_TOO_SMALL;
    }

    size_t needed_segments = 0;
    for (size_t qi = 0; qi < query_count; ++qi) {
        const auto & query = queries[qi];
        if (query.query_position < 0 || query.window_start > query.query_position) {
            return COMMON_TREE_DRAFT_KV_WINDOW_QUERY_RANGE;
        }

        uint64_t logical = 0;
        const uint32_t row_begin = source.row_ptr[qi];
        const uint32_t row_end = source.row_ptr[qi + 1];
        const uint64_t visible_begin = static_cast<uint64_t>(std::max<int64_t>(query.window_start, 0));
        const uint64_t visible_end = static_cast<uint64_t>(query.query_position) + 1;

        for (uint32_t si = row_begin; si < row_end; ++si) {
            const auto & segment = source.segments[si];
            const uint64_t len = static_cast<uint64_t>(segment.hi - segment.lo);
            if (logical > std::numeric_limits<uint64_t>::max() - len) {
                return COMMON_TREE_DRAFT_KV_WINDOW_OVERFLOW;
            }
            const uint64_t segment_end = logical + len;
            if (std::max(logical, visible_begin) < std::min(segment_end, visible_end)) {
                ++needed_segments;
            }
            logical = segment_end;
        }
        if (logical != visible_end) return COMMON_TREE_DRAFT_KV_WINDOW_QUERY_RANGE;
    }

    if (needed_segments > UINT32_MAX) return COMMON_TREE_DRAFT_KV_WINDOW_OVERFLOW;
    if (needed_segments > output->segment_capacity || (needed_segments > 0 && output->segments == nullptr)) {
        return COMMON_TREE_DRAFT_KV_WINDOW_OUTPUT_TOO_SMALL;
    }

    uint32_t cursor = 0;
    output->row_ptr[0] = 0;
    for (size_t qi = 0; qi < query_count; ++qi) {
        const auto & query = queries[qi];
        uint64_t logical = 0;
        const uint64_t visible_begin = static_cast<uint64_t>(std::max<int64_t>(query.window_start, 0));
        const uint64_t visible_end = static_cast<uint64_t>(query.query_position) + 1;

        for (uint32_t si = source.row_ptr[qi]; si < source.row_ptr[qi + 1]; ++si) {
            const auto & segment = source.segments[si];
            const uint64_t len = static_cast<uint64_t>(segment.hi - segment.lo);
            const uint64_t segment_end = logical + len;
            const uint64_t clipped_begin = std::max(logical, visible_begin);
            const uint64_t clipped_end = std::min(segment_end, visible_end);
            if (clipped_begin < clipped_end) {
                const uint64_t lo_delta = clipped_begin - logical;
                const uint64_t hi_delta = clipped_end - logical;
                if (lo_delta > UINT32_MAX || hi_delta > UINT32_MAX) {
                    return COMMON_TREE_DRAFT_KV_WINDOW_OVERFLOW;
                }
                output->segments[cursor++] = {
                    segment.page,
                    static_cast<uint32_t>(segment.lo + lo_delta),
                    static_cast<uint32_t>(segment.lo + hi_delta),
                };
            }
            logical = segment_end;
        }
        output->row_ptr[qi + 1] = cursor;
    }

    output->query_count = static_cast<uint32_t>(query_count);
    output->segment_count = cursor;
    return COMMON_TREE_DRAFT_KV_WINDOW_OK;
}
