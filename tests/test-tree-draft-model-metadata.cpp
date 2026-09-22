#include "tree-draft-model-metadata.h"

#include <cassert>
#include <cstdint>
#include <cstring>

static common_tree_draft_gguf_metadata_value string_value(const char * value) {
    common_tree_draft_gguf_metadata_value result;
    result.type = GGUF_TYPE_STRING;
    result.strings = { value };
    return result;
}

static common_tree_draft_gguf_metadata_value u32_value(uint32_t value) {
    common_tree_draft_gguf_metadata_value result;
    result.type = GGUF_TYPE_UINT32;
    result.raw.resize(sizeof(value));
    std::memcpy(result.raw.data(), &value, sizeof(value));
    return result;
}

static common_tree_draft_gguf_metadata_value f32_value(float value) {
    common_tree_draft_gguf_metadata_value result;
    result.type = GGUF_TYPE_FLOAT32;
    result.raw.resize(sizeof(value));
    std::memcpy(result.raw.data(), &value, sizeof(value));
    return result;
}

static void test_gguf_projection_preserves_provenance_and_unknown() {
    common_tree_draft_gguf_metadata_table table;
    table.entries = {
        { "general.architecture", string_value("llama") },
        { "llama.block_count", u32_value(28) },
        { "llama.embedding_length", u32_value(3072) },
        { "llama.attention.head_count", u32_value(24) },
        { "llama.rope.freq_base", f32_value(10000.0f) },
        { "vendor.extra", u32_value(7) },
    };

    const auto result = common_tree_draft_model_metadata_normalize(table);
    assert(result.architecture && result.architecture->value == "llama");
    assert(result.architecture->origin.format == COMMON_TREE_DRAFT_MODEL_FORMAT_GGUF);
    assert(result.architecture->origin.key == "general.architecture");
    assert(!result.architecture->origin.inferred);
    assert(result.n_layers && result.n_layers->value == 28);
    assert(result.n_embd && result.n_embd->value == 3072);
    assert(result.n_heads && result.n_heads->value == 24);
    assert(result.rope_theta && result.rope_theta->value == 10000.0);
    assert(!result.vocab_size);
    assert(result.unknown.size() == 1);
    assert(result.unknown[0].origin.key == "vendor.extra");
    assert(result.unknown[0].raw.size() == sizeof(uint32_t));
}

static void test_safetensors_projection_keeps_absence_distinct() {
    common_tree_draft_safetensors_table table;
    table.metadata = {
        { "architecture", "gemma4" },
        { "hidden_size", "4096" },
        { "num_hidden_layers", "34" },
        { "rope_theta", "1000000" },
        { "vendor.note", "keep" },
    };

    const auto result = common_tree_draft_model_metadata_normalize(table);
    assert(result.architecture && result.architecture->value == "gemma4");
    assert(result.architecture->origin.format == COMMON_TREE_DRAFT_MODEL_FORMAT_SAFETENSORS);
    assert(result.n_embd && result.n_embd->value == 4096);
    assert(result.n_layers && result.n_layers->value == 34);
    assert(result.rope_theta && result.rope_theta->value == 1000000.0);
    assert(!result.n_heads);
    assert(!result.n_heads_kv);
    assert(!result.vocab_size);
    assert(result.unknown.size() == 1);
    assert(result.unknown[0].origin.key == "vendor.note");
    assert(result.unknown[0].strings == std::vector<std::string>({ "keep" }));
}

static void test_malformed_known_gguf_value_is_preserved() {
    common_tree_draft_gguf_metadata_table table;
    table.entries = {
        { "general.architecture", string_value("llama") },
        { "general.quantization_version", string_value("not-an-integer") },
    };

    const auto result = common_tree_draft_model_metadata_normalize(table);
    assert(result.architecture && result.architecture->value == "llama");
    assert(!result.quantization_version);
    assert(result.unknown.size() == 1);
    assert(result.unknown[0].origin.key == "general.quantization_version");
    assert(result.unknown[0].strings == std::vector<std::string>({ "not-an-integer" }));
}

int main() {
    test_gguf_projection_preserves_provenance_and_unknown();
    test_safetensors_projection_keeps_absence_distinct();
    test_malformed_known_gguf_value_is_preserved();
    return 0;
}
