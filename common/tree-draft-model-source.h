#pragma once

#include <cstdint>
#include <string>
#include <vector>

enum common_tree_draft_model_format : int32_t {
    COMMON_TREE_DRAFT_MODEL_FORMAT_UNKNOWN = 0,
    COMMON_TREE_DRAFT_MODEL_FORMAT_GGUF,
    COMMON_TREE_DRAFT_MODEL_FORMAT_SAFETENSORS,
};

enum common_tree_draft_model_loader : int32_t {
    COMMON_TREE_DRAFT_MODEL_LOADER_NONE = 0,
    COMMON_TREE_DRAFT_MODEL_LOADER_GGUF,
    COMMON_TREE_DRAFT_MODEL_LOADER_SAFETENSORS,
};

enum common_tree_draft_model_probe_signal : int32_t {
    COMMON_TREE_DRAFT_MODEL_PROBE_FILE = 0,
    COMMON_TREE_DRAFT_MODEL_PROBE_DIRECTORY,
    COMMON_TREE_DRAFT_MODEL_PROBE_SAFETENSORS_INDEX,
    COMMON_TREE_DRAFT_MODEL_PROBE_GGUF_HEADER,
    COMMON_TREE_DRAFT_MODEL_PROBE_SAFETENSORS_HEADER,
    COMMON_TREE_DRAFT_MODEL_PROBE_MISSING,
    COMMON_TREE_DRAFT_MODEL_PROBE_INVALID,
    COMMON_TREE_DRAFT_MODEL_PROBE_AMBIGUOUS,
};

struct common_tree_draft_model_probe_evidence {
    std::string path;
    common_tree_draft_model_probe_signal signal = COMMON_TREE_DRAFT_MODEL_PROBE_INVALID;
    uint64_t file_size = 0;
    uint64_t bytes_read = 0;
    uint64_t header_size = 0;
};

struct common_tree_draft_model_source {
    common_tree_draft_model_format format = COMMON_TREE_DRAFT_MODEL_FORMAT_UNKNOWN;
    common_tree_draft_model_loader loader = COMMON_TREE_DRAFT_MODEL_LOADER_NONE;
    float confidence = 0.0f;
    std::vector<std::string> paths;
    std::vector<common_tree_draft_model_probe_evidence> evidence;
};

common_tree_draft_model_source common_tree_draft_model_source_probe(const std::string & input);
common_tree_draft_model_source common_tree_draft_model_source_probe(const std::vector<std::string> & inputs);

const char * common_tree_draft_model_format_name(common_tree_draft_model_format format);
const char * common_tree_draft_model_loader_name(common_tree_draft_model_loader loader);
const char * common_tree_draft_model_probe_signal_name(common_tree_draft_model_probe_signal signal);
