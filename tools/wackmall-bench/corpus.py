#!/usr/bin/env python3

import hashlib
import importlib.util
import json
from collections.abc import Callable, Mapping
from pathlib import Path, PurePosixPath


CORPUS_SCHEMA_VERSION = 1
CORPUS_FIELDS = ("schema_version", "corpus_id", "tokenizer", "prompts")
TOKENIZER_FIELDS = ("name", "sha256")
PROMPT_FIELDS = ("id", "category", "file", "sha256", "token_length")
PROMPT_CATEGORIES = frozenset(("short", "medium", "long", "code", "prose", "adversarial"))


class CorpusValidationError(ValueError):
    pass


def _is_sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def _require_exact_fields(value, expected, path):
    if not isinstance(value, Mapping):
        raise CorpusValidationError(f"{path} must be an object")
    keys = set(value)
    expected_keys = set(expected)
    missing = sorted(expected_keys - keys)
    extra = sorted(keys - expected_keys)
    if missing:
        raise CorpusValidationError(f"missing {path} fields: {', '.join(missing)}")
    if extra:
        raise CorpusValidationError(f"unknown {path} fields: {', '.join(extra)}")


def _validate_relative_file(value, path):
    if not isinstance(value, str) or not value:
        raise CorpusValidationError(f"{path} must be a non-empty relative path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or "\\" in value:
        raise CorpusValidationError(f"{path} must be a portable relative path without '..'")


def validate_corpus_manifest(manifest):
    _require_exact_fields(manifest, CORPUS_FIELDS, "corpus manifest")
    if manifest["schema_version"] != CORPUS_SCHEMA_VERSION:
        raise CorpusValidationError(f"schema_version must be {CORPUS_SCHEMA_VERSION}")

    corpus_id = manifest["corpus_id"]
    if not isinstance(corpus_id, str) or not corpus_id.strip():
        raise CorpusValidationError("corpus_id must be a non-empty string")

    tokenizer = manifest["tokenizer"]
    _require_exact_fields(tokenizer, TOKENIZER_FIELDS, "tokenizer")
    if not isinstance(tokenizer["name"], str) or not tokenizer["name"].strip():
        raise CorpusValidationError("tokenizer.name must be a non-empty string")
    if not _is_sha256(tokenizer["sha256"]):
        raise CorpusValidationError("tokenizer.sha256 must be a lowercase SHA-256 hex digest")

    prompts = manifest["prompts"]
    if not isinstance(prompts, list) or not prompts:
        raise CorpusValidationError("prompts must be a non-empty array")

    seen_ids = set()
    for index, prompt in enumerate(prompts):
        path = f"prompts[{index}]"
        _require_exact_fields(prompt, PROMPT_FIELDS, path)
        prompt_id = prompt["id"]
        if not isinstance(prompt_id, str) or not prompt_id.strip():
            raise CorpusValidationError(f"{path}.id must be a non-empty string")
        if prompt_id in seen_ids:
            raise CorpusValidationError(f"duplicate prompt id: {prompt_id}")
        seen_ids.add(prompt_id)
        if prompt["category"] not in PROMPT_CATEGORIES:
            raise CorpusValidationError(f"{path}.category is not recognized")
        _validate_relative_file(prompt["file"], f"{path}.file")
        if not _is_sha256(prompt["sha256"]):
            raise CorpusValidationError(f"{path}.sha256 must be a lowercase SHA-256 hex digest")
        token_length = prompt["token_length"]
        if isinstance(token_length, bool) or not isinstance(token_length, int) or token_length <= 0:
            raise CorpusValidationError(f"{path}.token_length must be a positive integer")

    return manifest


def ordered_pairs(manifest):
    validate_corpus_manifest(manifest)
    return tuple((prompt["sha256"], prompt["token_length"]) for prompt in manifest["prompts"])


def corpus_identity(manifest):
    validate_corpus_manifest(manifest)
    identity = {
        "schema_version": CORPUS_SCHEMA_VERSION,
        "tokenizer": manifest["tokenizer"],
        "ordered_pairs": ordered_pairs(manifest),
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def verify_prompt_files(manifest, manifest_path, token_length: Callable[[bytes], int] | None = None):
    validate_corpus_manifest(manifest)
    base = Path(manifest_path).resolve().parent
    for prompt in manifest["prompts"]:
        prompt_path = (base / PurePosixPath(prompt["file"])).resolve()
        try:
            prompt_path.relative_to(base)
        except ValueError as exc:
            raise CorpusValidationError(f"prompt file escapes manifest directory: {prompt['file']}") from exc
        data = prompt_path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != prompt["sha256"]:
            raise CorpusValidationError(f"prompt hash mismatch for {prompt['id']}")
        if token_length is not None:
            actual_length = token_length(data)
            if actual_length != prompt["token_length"]:
                raise CorpusValidationError(f"token length mismatch for {prompt['id']}")
    return manifest


def load_corpus_manifest(path, token_length: Callable[[bytes], int] | None = None):
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    validate_corpus_manifest(manifest)
    verify_prompt_files(manifest, path, token_length=token_length)
    return manifest


def _load_scenario_module():
    module_path = Path(__file__).with_name("scenario.py")
    spec = importlib.util.spec_from_file_location("wackmall_bench_scenario_for_corpus", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bind_scenario(manifest, scenario):
    validate_corpus_manifest(manifest)
    scenario_module = _load_scenario_module()
    scenario_module.validate_scenario(scenario)
    if scenario["prompt_set"] != manifest["corpus_id"]:
        raise CorpusValidationError("scenario prompt_set does not match corpus_id")
    return {
        "scenario_hash": scenario_module.scenario_hash(scenario),
        "corpus_id": manifest["corpus_id"],
        "corpus_identity": corpus_identity(manifest),
    }
