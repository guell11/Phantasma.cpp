import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("scenario.py")
SPEC = importlib.util.spec_from_file_location("wackmall_bench_scenario", MODULE_PATH)
SCENARIO = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCENARIO)
FIXTURES = Path(__file__).with_name("fixtures")


class ScenarioTests(unittest.TestCase):
    def test_fixtures_validate_and_hash_stably(self):
        expected_hashes = {
            "tree-draft-greedy.json": "64a20014d29af3760a8f8d4c259285c6d619cb4eeb81336d722f84e023383316",
            "tree-draft-sampled.json": "f23b017c4adb31ee0e83fb54b9cae08d8bfb7d5428b8e3c53af7ff540c406f86",
        }
        for name, expected_hash in expected_hashes.items():
            scenario = SCENARIO.load_scenario(FIXTURES / name)
            canonical = SCENARIO.canonical_json(scenario)
            reordered = dict(reversed(list(scenario.items())))
            self.assertEqual(SCENARIO.canonical_json(reordered), canonical)
            self.assertEqual(SCENARIO.scenario_hash(reordered), SCENARIO.scenario_hash(scenario))
            self.assertEqual(SCENARIO.scenario_hash(scenario), expected_hash)

    def test_hash_changes_with_semantic_field(self):
        scenario = SCENARIO.load_scenario(FIXTURES / "tree-draft-greedy.json")
        changed = dict(scenario)
        changed["seed"] += 1
        self.assertNotEqual(SCENARIO.scenario_hash(changed), SCENARIO.scenario_hash(scenario))

    def test_nested_object_order_is_canonical(self):
        scenario = SCENARIO.load_scenario(FIXTURES / "tree-draft-sampled.json")
        changed = dict(scenario)
        changed["sampling"] = dict(reversed(list(scenario["sampling"].items())))
        self.assertEqual(SCENARIO.canonical_bytes(changed), SCENARIO.canonical_bytes(scenario))

    def test_validation_rejects_missing_unknown_and_invalid_values(self):
        scenario = SCENARIO.load_scenario(FIXTURES / "tree-draft-greedy.json")

        missing = dict(scenario)
        del missing["device"]
        with self.assertRaisesRegex(SCENARIO.ScenarioValidationError, "missing scenario fields"):
            SCENARIO.validate_scenario(missing)

        extra = dict(scenario, repetitions=3)
        with self.assertRaisesRegex(SCENARIO.ScenarioValidationError, "unknown scenario fields"):
            SCENARIO.validate_scenario(extra)

        invalid = dict(scenario, batch=0)
        with self.assertRaisesRegex(SCENARIO.ScenarioValidationError, "batch must be a positive integer"):
            SCENARIO.validate_scenario(invalid)

        invalid = dict(scenario, seed=-1)
        with self.assertRaisesRegex(SCENARIO.ScenarioValidationError, "unsigned 64-bit"):
            SCENARIO.validate_scenario(invalid)

        invalid = dict(scenario)
        invalid["sampling"] = {"temperature": float("nan")}
        with self.assertRaisesRegex(SCENARIO.ScenarioValidationError, "NaN or infinity"):
            SCENARIO.validate_scenario(invalid)

    def test_loader_rejects_non_object(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "scenario.json"
            path.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")
            with self.assertRaisesRegex(SCENARIO.ScenarioValidationError, "scenario must be an object"):
                SCENARIO.load_scenario(path)

    def test_all_semantic_fields_change_identity(self):
        scenario = SCENARIO.load_scenario(FIXTURES / "tree-draft-greedy.json")
        baseline = SCENARIO.scenario_hash(scenario)
        changes = {
            "model": "other-target.gguf",
            "draft_model": "other-draft.gguf",
            "prompt_set": "other-prompts",
            "batch": 2,
            "n_ctx": 8192,
            "tree_shape": {"depth": 3, "branching": [3, 2, 1]},
            "sampling": {"temperature": 0.5, "top_k": 20, "top_p": 0.9},
            "seed": 2,
            "backend": "CPU",
            "device": "1",
        }

        for field, value in changes.items():
            with self.subTest(field=field):
                changed = dict(scenario)
                changed[field] = value
                self.assertNotEqual(SCENARIO.scenario_hash(changed), baseline)

    def test_canonicalization_ignores_mapping_insertion_order_only(self):
        scenario = SCENARIO.load_scenario(FIXTURES / "tree-draft-sampled.json")
        reordered = {field: scenario[field] for field in reversed(SCENARIO.SCENARIO_FIELDS)}
        reordered["tree_shape"] = dict(reversed(list(scenario["tree_shape"].items())))
        reordered["sampling"] = dict(reversed(list(scenario["sampling"].items())))

        self.assertEqual(SCENARIO.canonical_bytes(reordered), SCENARIO.canonical_bytes(scenario))
        self.assertEqual(SCENARIO.scenario_hash(reordered), SCENARIO.scenario_hash(scenario))

    def test_validation_accepts_integer_boundaries(self):
        scenario = SCENARIO.load_scenario(FIXTURES / "tree-draft-greedy.json")

        for seed in (0, 0xFFFFFFFFFFFFFFFF):
            with self.subTest(seed=seed):
                accepted = dict(scenario, batch=1, n_ctx=1, seed=seed)
                self.assertIs(SCENARIO.validate_scenario(accepted), accepted)

    def test_validation_rejects_invalid_scalar_ranges_and_types(self):
        scenario = SCENARIO.load_scenario(FIXTURES / "tree-draft-greedy.json")
        cases = (
            ("batch", False, "positive integer"),
            ("batch", -1, "positive integer"),
            ("n_ctx", True, "positive integer"),
            ("n_ctx", 0, "positive integer"),
            ("seed", True, "unsigned 64-bit"),
            ("seed", -1, "unsigned 64-bit"),
            ("seed", 0x10000000000000000, "unsigned 64-bit"),
            ("model", "", "non-empty string"),
            ("draft_model", "   ", "non-empty string"),
            ("prompt_set", None, "non-empty string"),
            ("backend", 1, "non-empty string"),
            ("device", "", "non-empty string"),
        )

        for field, value, error in cases:
            with self.subTest(field=field, value=value):
                invalid = dict(scenario)
                invalid[field] = value
                with self.assertRaisesRegex(SCENARIO.ScenarioValidationError, error):
                    SCENARIO.validate_scenario(invalid)

    def test_validation_rejects_invalid_nested_contract(self):
        scenario = SCENARIO.load_scenario(FIXTURES / "tree-draft-greedy.json")
        cases = (
            ("tree_shape", {}, "non-empty object"),
            ("tree_shape", [], "non-empty object"),
            ("sampling", {}, "non-empty object"),
            ("sampling", [], "non-empty object"),
            ("sampling", {"temperature": float("inf")}, "NaN or infinity"),
            ("sampling", {"temperature": float("-inf")}, "NaN or infinity"),
            ("tree_shape", {1: "bad-key"}, "keys must be non-empty strings"),
            ("tree_shape", {"": "bad-key"}, "keys must be non-empty strings"),
            ("tree_shape", {"depth": object()}, "unsupported value type"),
        )

        for field, value, error in cases:
            with self.subTest(field=field, value=value):
                invalid = dict(scenario)
                invalid[field] = value
                with self.assertRaisesRegex(SCENARIO.ScenarioValidationError, error):
                    SCENARIO.validate_scenario(invalid)


if __name__ == "__main__":
    unittest.main()
