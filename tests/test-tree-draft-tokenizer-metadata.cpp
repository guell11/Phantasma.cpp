#include "tree-draft-tokenizer-metadata.h"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <utility>

static common_tree_draft_model_metadata_unknown text_unknown(
        common_tree_draft_model_format format,
        const char * key,
        std::vector<std::string> values) {
    common_tree_draft_model_metadata_unknown result;
    result.origin = { format, key, 1.0f, false };
    result.strings = std::move(values);
    return result;
}

static common_tree_draft_model_metadata_unknown u32_unknown(
        common_tree_draft_model_format format,
        const char * key,
        uint32_t value) {
    common_tree_draft_model_metadata_unknown result;
    result.origin = { format, key, 1.0f, false };
    result.gguf_type = GGUF_TYPE_UINT32;
    result.raw.resize(sizeof(value));
    std::memcpy(result.raw.data(), &value, sizeof(value));
    return result;
}

static common_tree_draft_model_metadata complete_gguf_metadata() {
    common_tree_draft_model_metadata metadata;
    metadata.vocab_size = { 4, { COMMON_TREE_DRAFT_MODEL_FORMAT_GGUF, "llama.vocab_size", 1.0f, false } };
    metadata.unknown = {
        text_unknown(COMMON_TREE_DRAFT_MODEL_FORMAT_GGUF, "tokenizer.ggml.model", { "gpt2" }),
        text_unknown(COMMON_TREE_DRAFT_MODEL_FORMAT_GGUF, "tokenizer.ggml.tokens", { "a", "b", "c", "d" }),
        text_unknown(COMMON_TREE_DRAFT_MODEL_FORMAT_GGUF, "tokenizer.ggml.merges", { "a b", "c d" }),
        u32_unknown(COMMON_TREE_DRAFT_MODEL_FORMAT_GGUF, "tokenizer.ggml.eos_token_id", 2),
        u32_unknown(COMMON_TREE_DRAFT_MODEL_FORMAT_GGUF, "tokenizer.ggml.bos_token_id", 1),
    };
    return metadata;
}

static void test_complete_signature_is_stable_and_exact() {
    const auto metadata = complete_gguf_metadata();
    const auto first = common_tree_draft_tokenizer_metadata_normalize(metadata);
    const auto second = common_tree_draft_tokenizer_metadata_normalize(metadata);

    assert(first.tokenizer_family && *first.tokenizer_family == "gpt2");
    assert(first.vocab_size && *first.vocab_size == 4);
    assert(first.vocab_hash && second.vocab_hash && first.vocab_hash == second.vocab_hash);
    assert(first.merge_or_model_hash && second.merge_or_model_hash && first.merge_or_model_hash == second.merge_or_model_hash);
    assert(first.has_special_token_metadata);
    assert(first.special_tokens.size() == 2);
    assert(first.special_tokens[0].role == "bos" && first.special_tokens[0].id == 1);
    assert(first.special_tokens[1].role == "eos" && first.special_tokens[1].id == 2);

    const auto compatibility = common_tree_draft_tokenizer_compare(first, second);
    assert(compatibility.status == COMMON_TREE_DRAFT_TOKENIZER_EXACT);
    assert(compatibility.mismatched_fields.empty());
    assert(compatibility.missing_fields.empty());
}

static void test_partial_metadata_stays_partial() {
    common_tree_draft_model_metadata metadata;
    metadata.vocab_size = { 32000, { COMMON_TREE_DRAFT_MODEL_FORMAT_SAFETENSORS, "vocab_size", 1.0f, false } };
    metadata.unknown = {
        text_unknown(COMMON_TREE_DRAFT_MODEL_FORMAT_SAFETENSORS, "tokenizer_class", { "LlamaTokenizer" }),
        text_unknown(COMMON_TREE_DRAFT_MODEL_FORMAT_SAFETENSORS, "vocab_hash", { "abc123" }),
    };

    const auto lhs = common_tree_draft_tokenizer_metadata_normalize(metadata);
    const auto rhs = common_tree_draft_tokenizer_metadata_normalize(metadata);
    const auto compatibility = common_tree_draft_tokenizer_compare(lhs, rhs);
    assert(compatibility.status == COMMON_TREE_DRAFT_TOKENIZER_PARTIAL);
    assert(compatibility.mismatched_fields.empty());
    assert(!compatibility.missing_fields.empty());
}

static void test_known_mismatch_is_incompatible() {
    auto lhs_metadata = complete_gguf_metadata();
    auto rhs_metadata = complete_gguf_metadata();
    rhs_metadata.unknown[1] = text_unknown(
            COMMON_TREE_DRAFT_MODEL_FORMAT_GGUF,
            "tokenizer.ggml.tokens",
            { "a", "b", "x", "d" });

    const auto lhs = common_tree_draft_tokenizer_metadata_normalize(lhs_metadata);
    const auto rhs = common_tree_draft_tokenizer_metadata_normalize(rhs_metadata);
    const auto compatibility = common_tree_draft_tokenizer_compare(lhs, rhs);
    assert(compatibility.status == COMMON_TREE_DRAFT_TOKENIZER_INCOMPATIBLE);
    assert(compatibility.mismatched_fields == std::vector<std::string>({ "vocab_hash" }));
}

static void test_token_array_can_supply_vocab_size() {
    common_tree_draft_model_metadata metadata;
    metadata.unknown = {
        text_unknown(COMMON_TREE_DRAFT_MODEL_FORMAT_GGUF, "tokenizer.ggml.tokens", { "x", "y", "z" }),
    };
    const auto signature = common_tree_draft_tokenizer_metadata_normalize(metadata);
    assert(signature.vocab_size && *signature.vocab_size == 3);
    assert(signature.vocab_hash);
}

static void test_malformed_special_token_is_not_exact() {
    auto metadata = complete_gguf_metadata();
    metadata.unknown.back() = text_unknown(
            COMMON_TREE_DRAFT_MODEL_FORMAT_GGUF,
            "tokenizer.ggml.bos_token_id",
            { "bad" });

    const auto signature = common_tree_draft_tokenizer_metadata_normalize(metadata);
    const auto compatibility = common_tree_draft_tokenizer_compare(signature, signature);
    assert(signature.has_special_token_metadata);
    assert(!signature.special_token_metadata_valid);
    assert(compatibility.status == COMMON_TREE_DRAFT_TOKENIZER_PARTIAL);
}

int main() {
    test_complete_signature_is_stable_and_exact();
    test_partial_metadata_stays_partial();
    test_known_mismatch_is_incompatible();
    test_token_array_can_supply_vocab_size();
    test_malformed_special_token_is_not_exact();
    return 0;
}
