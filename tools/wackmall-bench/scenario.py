#!/usr/bin/env python3

import hashlib
import json
import math
from collections.abc import Mapping


SCENARIO_FIELDS = (
    "model",
    "draft_model",
    "prompt_set",
    "batch",
    "n_ctx",
    "tree_shape",
    "sampling",
    "seed",
    "backend",
    "device",
)


class ScenarioValidationError(ValueError):
    pass


def _validate_json_value(value, path):
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ScenarioValidationError(f"{path} must not contain NaN or infinity")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{path}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ScenarioValidationError(f"{path} keys must be non-empty strings")
            _validate_json_value(item, f"{path}.{key}")
        return
    raise ScenarioValidationError(f"{path} contains unsupported value type {type(value).__name__}")


def validate_scenario(scenario):
    if not isinstance(scenario, Mapping):
        raise ScenarioValidationError("scenario must be an object")

    keys = set(scenario)
    expected = set(SCENARIO_FIELDS)
    missing = sorted(expected - keys)
    extra = sorted(keys - expected)
    if missing:
        raise ScenarioValidationError(f"missing scenario fields: {', '.join(missing)}")
    if extra:
        raise ScenarioValidationError(f"unknown scenario fields: {', '.join(extra)}")

    for name in ("model", "draft_model", "prompt_set", "backend", "device"):
        value = scenario[name]
        if not isinstance(value, str) or not value.strip():
            raise ScenarioValidationError(f"{name} must be a non-empty string")

    for name in ("batch", "n_ctx"):
        value = scenario[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ScenarioValidationError(f"{name} must be a positive integer")

    seed = scenario["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 0xFFFFFFFFFFFFFFFF:
        raise ScenarioValidationError("seed must be an unsigned 64-bit integer")

    for name in ("tree_shape", "sampling"):
        value = scenario[name]
        if not isinstance(value, Mapping) or not value:
            raise ScenarioValidationError(f"{name} must be a non-empty object")
        _validate_json_value(value, name)

    return scenario


def canonical_bytes(scenario):
    validate_scenario(scenario)
    normalized = {field: scenario[field] for field in SCENARIO_FIELDS}
    try:
        text = json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ScenarioValidationError(str(exc)) from exc
    return text.encode("utf-8")


def canonical_json(scenario):
    return canonical_bytes(scenario).decode("utf-8")


def scenario_hash(scenario):
    return hashlib.sha256(canonical_bytes(scenario)).hexdigest()


def load_scenario(path):
    with open(path, "r", encoding="utf-8") as handle:
        scenario = json.load(handle)
    validate_scenario(scenario)
    return scenario
