find_package(Threads REQUIRED)

llama_add_compile_flags()

#
# llama-common-base
#

# Build info header

if(EXISTS "${PROJECT_SOURCE_DIR}/.git")
    set(GIT_DIR "${PROJECT_SOURCE_DIR}/.git")

    # Is git submodule
    if(NOT IS_DIRECTORY "${GIT_DIR}")
        file(READ ${GIT_DIR} REAL_GIT_DIR_LINK)
        string(REGEX REPLACE "gitdir: (.*)\n$" "\\1" REAL_GIT_DIR ${REAL_GIT_DIR_LINK})
        string(FIND "${REAL_GIT_DIR}" "/" SLASH_POS)
        if (SLASH_POS EQUAL 0)
            set(GIT_DIR "${REAL_GIT_DIR}")
        else()
            set(GIT_DIR "${PROJECT_SOURCE_DIR}/${REAL_GIT_DIR}")
        endif()
    endif()

    if(EXISTS "${GIT_DIR}/index")
        # For build-info.cpp below
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${GIT_DIR}/index")
    else()
        message(WARNING "Git index not found in git repository.")
    endif()
else()
    message(WARNING "Git repository not found; to enable automatic generation of build info, make sure Git is installed and the project is a Git repository.")
endif()

set(TEMPLATE_FILE "${CMAKE_CURRENT_SOURCE_DIR}/build-info.cpp.in")
set(OUTPUT_FILE   "${CMAKE_CURRENT_BINARY_DIR}/build-info.cpp")

configure_file(${TEMPLATE_FILE} ${OUTPUT_FILE})

set(TARGET llama-common-base)
add_library(${TARGET} STATIC ${OUTPUT_FILE})

target_include_directories(${TARGET} PUBLIC .)

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

#
# llama-common
#

set(TARGET llama-common)

add_library(${TARGET}
    arg.cpp
    arg.h
    base64.hpp
    chat-auto-parser-generator.cpp
    chat-auto-parser-helpers.cpp
    chat-auto-parser.h
    chat-diff-analyzer.cpp
    chat-peg-parser.cpp
    chat-peg-parser.h
    chat.cpp
    chat.h
    common.cpp
    common.h
    console.cpp
    console.h
    debug.cpp
    debug.h
    download.cpp
    download.h
    fit.cpp
    fit.h
    hf-cache.cpp
    hf-cache.h
    http.h
    imatrix-loader.cpp
    imatrix-loader.h
    json-schema-to-grammar.cpp
    llguidance.cpp
    log.cpp
    log.h
    ngram-cache.cpp
    ngram-cache.h
    ngram-map.cpp
    ngram-map.h
    ngram-mod.cpp
    ngram-mod.h
    peg-parser.cpp
    peg-parser.h
    preset.cpp
    preset.h
    reasoning-budget.cpp
    reasoning-budget.h
    sampling.cpp
    sampling.h
    speculative.cpp
    speculative.h
    tree-draft-acceptance.cpp
    tree-draft-acceptance.h
    tree-draft-alignment.cpp
    tree-draft-alignment.h
    tree-draft-abi.cpp
    tree-draft-abi.h
    tree-draft-ancestor.cpp
    tree-draft-ancestor.h
    tree-draft-ancestor-format.cpp
    tree-draft-ancestor-format.h
    tree-draft-async-staging.cpp
    tree-draft-async-staging.h
    tree-draft-attention-oracle.cpp
    tree-draft-attention-oracle.h
    tree-draft-attention-baseline.cpp
    tree-draft-attention-baseline.h
    tree-draft-append.cpp
    tree-draft-append.h
    tree-draft-arena.cpp
    tree-draft-arena.h
    tree-draft-budget.cpp
    tree-draft-budget.h
    tree-draft-buffered-load.cpp
    tree-draft-buffered-load.h
    tree-draft-composite-mask.cpp
    tree-draft-composite-mask.h
    tree-draft-architecture-config.cpp
    tree-draft-architecture-config.h
    tree-draft-c-api.cpp
    tree-draft-c-api.h
    tree-draft-categorical.cpp
    tree-draft-categorical.h
    tree-draft-candidates.cpp
    tree-draft-candidates.h
    tree-draft-coalescing.cpp
    tree-draft-coalescing.h
    tree-draft-child-budget.cpp
    tree-draft-child-budget.h
    tree-draft-child-write.cpp
    tree-draft-child-write.h
    tree-draft-dense-mask.cpp
    tree-draft-dense-mask.h
    tree-draft-dequant.cpp
    tree-draft-dequant.h
    tree-draft-dequant-fallback.cpp
    tree-draft-dequant-fallback.h
    tree-draft-dequant-plan.cpp
    tree-draft-dequant-plan.h
    tree-draft-device-allocator.cpp
    tree-draft-device-allocator.h
    tree-draft-error.cpp
    tree-draft-error.h
    tree-draft-engine-lifecycle.cpp
    tree-draft-engine-lifecycle.h
    tree-draft-external-quant.cpp
    tree-draft-external-quant.h
    tree-draft-config.cpp
    tree-draft-config.h
    tree-draft-forest.cpp
    tree-draft-forest.h
    tree-draft-frontier-prune.cpp
    tree-draft-frontier-prune.h
    tree-draft-fused-dequant-staging.cpp
    tree-draft-fused-dequant-staging.h
    tree-draft-gguf-header.cpp
    tree-draft-gguf-header.h
    tree-draft-gguf-metadata.cpp
    tree-draft-gguf-metadata.h
    tree-draft-gguf-mmap-plan.cpp
    tree-draft-gguf-mmap-plan.h
    tree-draft-gguf-shards.cpp
    tree-draft-gguf-shards.h
    tree-draft-gguf-tensors.cpp
    tree-draft-gguf-tensors.h
    tree-draft-gptq-reindex.cpp
    tree-draft-gptq-reindex.h
    tree-draft-head-map.cpp
    tree-draft-head-map.h
    tree-draft-handle.cpp
    tree-draft-handle.h
    tree-draft-host.h
    tree-draft-kv-compat.cpp
    tree-draft-kv-compat.h
    tree-draft-kv-branch.cpp
    tree-draft-kv-branch.h
    tree-draft-kv-block-table.cpp
    tree-draft-kv-block-table.h
    tree-draft-kv-head-map.cpp
    tree-draft-kv-head-map.h
    tree-draft-kv-gather.cpp
    tree-draft-kv-gather.h
    tree-draft-kv-mask.cpp
    tree-draft-kv-mask.h
    tree-draft-kv-window.cpp
    tree-draft-kv-window.h
    tree-draft-kv-identity.cpp
    tree-draft-kv-identity.h
    tree-draft-kv-page.cpp
    tree-draft-kv-page.h
    tree-draft-kv-quant-page.cpp
    tree-draft-kv-quant-page.h
    tree-draft-kv-page-table.cpp
    tree-draft-kv-page-table.h
    tree-draft-kv-visibility.cpp
    tree-draft-kv-visibility.h
    tree-draft-kernel-capability.cpp
    tree-draft-kernel-capability.h
    tree-draft-loader-error.cpp
    tree-draft-loader-error.h
    tree-draft-kv-indirection.cpp
    tree-draft-kv-indirection.h
    tree-draft-lifecycle.cpp
    tree-draft-lifecycle.h
    tree-draft-integrity.cpp
    tree-draft-integrity.h
    tree-draft-mapping-owner.cpp
    tree-draft-mapping-owner.h
    tree-draft-mass-prune.cpp
    tree-draft-mass-prune.h
    tree-draft-memory-mode.cpp
    tree-draft-memory-mode.h
    tree-draft-model-source.cpp
    tree-draft-model-source.h
    tree-draft-model-metadata.cpp
    tree-draft-model-metadata.h
    tree-draft-moe-layout.cpp
    tree-draft-moe-layout.h
    tree-draft-multi-sample.cpp
    tree-draft-multi-sample.h
    tree-draft-node.cpp
    tree-draft-node.h
    tree-draft-next-frontier.cpp
    tree-draft-next-frontier.h
    tree-draft-online-softmax.cpp
    tree-draft-online-softmax.h
    tree-draft-path.cpp
    tree-draft-path.h
    tree-draft-packed-linear.cpp
    tree-draft-packed-linear.h
    tree-draft-tokenizer-metadata.cpp
    tree-draft-tokenizer-metadata.h
    tree-draft-position.cpp
    tree-draft-position.h
    tree-draft-precision.cpp
    tree-draft-precision.h
    tree-draft-qkv.cpp
    tree-draft-qkv.h
    tree-draft-ragged-attention.cpp
    tree-draft-ragged-attention.h
    tree-draft-repack-plan.cpp
    tree-draft-repack-plan.h
    tree-draft-ranking.cpp
    tree-draft-ranking.h
    tree-draft-quant-registry.cpp
    tree-draft-quant-registry.h
    tree-draft-quant-groups.cpp
    tree-draft-quant-groups.h
    tree-draft-quant-matmul-dispatch.cpp
    tree-draft-quant-matmul-dispatch.h
    tree-draft-quant-blocks.cpp
    tree-draft-quant-blocks.h
    tree-draft-runtime-status.cpp
    tree-draft-runtime-status.h
    tree-draft-safetensors.cpp
    tree-draft-safetensors.h
    tree-draft-safetensors-index.cpp
    tree-draft-safetensors-index.h
    tree-draft-sample.cpp
    tree-draft-sample.h
    tree-draft-sampling-rng.cpp
    tree-draft-sampling-rng.h
    tree-draft-score.cpp
    tree-draft-score.h
    tree-draft-scheduler.cpp
    tree-draft-scheduler.h
    tree-draft-sm89.cpp
    tree-draft-sm89.h
    tree-draft-smem-carveout.cpp
    tree-draft-smem-carveout.h
    tree-draft-smem-layout.cpp
    tree-draft-smem-layout.h
    tree-draft-storage-view.cpp
    tree-draft-storage-view.h
    tree-draft-storage-validate.cpp
    tree-draft-storage-validate.h
    tree-draft-tensor-role.cpp
    tree-draft-tensor-role.h
    tree-draft-tensor-shape.cpp
    tree-draft-tensor-shape.h
    tree-draft-tensor-layout.cpp
    tree-draft-tensor-layout.h
    tree-draft-tensor-core-policy.cpp
    tree-draft-tensor-core-policy.h
    tree-draft-top-k.cpp
    tree-draft-top-k.h
    tree-draft-top-p.cpp
    tree-draft-top-p.h
    tree-draft-tile-visibility.cpp
    tree-draft-tile-visibility.h
    tree-draft-topology.cpp
    tree-draft-topology.h
    tree-draft-validator.cpp
    tree-draft-validator.h
    tree-draft-visibility-decode.cpp
    tree-draft-visibility-decode.h
    tree-draft-width-policy.cpp
    tree-draft-width-policy.h
    subproc.cpp
    subproc.h
    trie.cpp
    trie.h
    unicode.cpp
    unicode.h
    jinja/lexer.cpp
    jinja/lexer.h
    jinja/parser.cpp
    jinja/parser.h
    jinja/runtime.cpp
    jinja/runtime.h
    jinja/value.cpp
    jinja/value.h
    jinja/string.cpp
    jinja/string.h
    jinja/caps.cpp
    jinja/caps.h
    )

set_target_properties(${TARGET} PROPERTIES
    VERSION ${LLAMA_INSTALL_VERSION}
    SOVERSION 0
    MACHO_CURRENT_VERSION 0 # keep macOS linker from seeing oversized version number
)

target_include_directories(${TARGET} PUBLIC . ../vendor)
target_compile_features   (${TARGET} PUBLIC cxx_std_17)

if (LLAMA_SUBPROCESS)
    target_compile_definitions(${TARGET} PUBLIC LLAMA_SUBPROCESS)
endif()

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)

    # TODO: make fine-grained exports in the future
    set_target_properties(${TARGET} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

target_link_libraries(${TARGET} PUBLIC  llama-common-base)
target_link_libraries(${TARGET} PRIVATE cpp-httplib)

if (LLAMA_LLGUIDANCE)
    include(ExternalProject)
    set(LLGUIDANCE_SRC ${CMAKE_BINARY_DIR}/llguidance/source)
    set(LLGUIDANCE_PATH ${LLGUIDANCE_SRC}/target/release)
    set(LLGUIDANCE_LIB_NAME "${CMAKE_STATIC_LIBRARY_PREFIX}llguidance${CMAKE_STATIC_LIBRARY_SUFFIX}")

    ExternalProject_Add(llguidance_ext
        GIT_REPOSITORY https://github.com/guidance-ai/llguidance
        # v1.0.1:
        GIT_TAG d795912fedc7d393de740177ea9ea761e7905774
        PREFIX ${CMAKE_BINARY_DIR}/llguidance
        SOURCE_DIR ${LLGUIDANCE_SRC}
        BUILD_IN_SOURCE TRUE
        CONFIGURE_COMMAND ""
        BUILD_COMMAND cargo build --release --package llguidance
        INSTALL_COMMAND ""
        BUILD_BYPRODUCTS ${LLGUIDANCE_PATH}/${LLGUIDANCE_LIB_NAME} ${LLGUIDANCE_PATH}/llguidance.h
        UPDATE_COMMAND ""
    )
    target_compile_definitions(${TARGET} PUBLIC LLAMA_USE_LLGUIDANCE)

    add_library(llguidance STATIC IMPORTED)
    set_target_properties(llguidance PROPERTIES IMPORTED_LOCATION ${LLGUIDANCE_PATH}/${LLGUIDANCE_LIB_NAME})
    add_dependencies(llguidance llguidance_ext)

    target_include_directories(${TARGET} PRIVATE ${LLGUIDANCE_PATH})
    target_link_libraries(${TARGET} PRIVATE llguidance)
    if (WIN32)
        target_link_libraries(${TARGET} PRIVATE ws2_32 userenv ntdll bcrypt)
    endif()
endif()

target_link_libraries(${TARGET} PUBLIC llama Threads::Threads)
