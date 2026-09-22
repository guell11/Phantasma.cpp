import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("corpus.py")
SPEC = importlib.util.spec_from_file_location("wackmall_bench_corpus", MODULE_PATH)
CORPUS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CORPUS)
FIXTURES = Path(__file__).with_name("fixtures")
MANIFEST = FIXTURES / "prompt-corpus-smoke-v1.json"


class CorpusTests(unittest.TestCase):
    def test_fixture_covers_categories_and_verifies_hashes_and_lengths(self):
        manifest = CORPUS.load_corpus_manifest(MANIFEST, token_length=len)
        self.assertEqual({prompt["category"] for prompt in manifest["prompts"]}, CORPUS.PROMPT_CATEGORIES)
        self.assertEqual(len(CORPUS.ordered_pairs(manifest)), 6)

    def test_identity_is_order_sensitive_and_metadata_insensitive(self):
        manifest = CORPUS.load_corpus_manifest(MANIFEST)
        expected = "bc4b6d813948c139e1bd798e23b7069f5fcfc22afb6d11ec5f30e1d5e5b621cb"
        self.assertEqual(CORPUS.corpus_identity(manifest), expected)

        reordered = copy.deepcopy(manifest)
        reordered["prompts"] = list(reversed(reordered["prompts"]))
        self.assertNotEqual(CORPUS.corpus_identity(reordered), expected)

        renamed = copy.deepcopy(manifest)
        renamed["corpus_id"] = "same-content-different-label"
        self.assertEqual(CORPUS.corpus_identity(renamed), expected)

    def test_scenario_binding_consumes_id_351_identity(self):
        manifest = CORPUS.load_corpus_manifest(MANIFEST)
        scenario_path = FIXTURES / "tree-draft-greedy.json"
        scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
        binding = CORPUS.bind_scenario(manifest, scenario)
        self.assertEqual(binding["corpus_id"], scenario["prompt_set"])
        self.assertEqual(binding["scenario_hash"], "64a20014d29af3760a8f8d4c259285c6d619cb4eeb81336d722f84e023383316")
        self.assertEqual(binding["corpus_identity"], CORPUS.corpus_identity(manifest))

        wrong = dict(scenario, prompt_set="other-v1")
        with self.assertRaisesRegex(CORPUS.CorpusValidationError, "prompt_set"):
            CORPUS.bind_scenario(manifest, wrong)

    def test_tampered_prompt_and_token_length_are_rejected(self):
        manifest = CORPUS.load_corpus_manifest(MANIFEST)
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            prompt = temp / "prompt.txt"
            prompt.write_bytes(b"abc")
            one = copy.deepcopy(manifest)
            one["prompts"] = [copy.deepcopy(one["prompts"][0])]
            one["prompts"][0].update({
                "file": "prompt.txt",
                "sha256": hashlib.sha256(b"abc").hexdigest(),
                "token_length": 4,
            })
            path = temp / "manifest.json"
            path.write_text(json.dumps(one), encoding="utf-8")
            with self.assertRaisesRegex(CORPUS.CorpusValidationError, "token length mismatch"):
                CORPUS.load_corpus_manifest(path, token_length=len)

            prompt.write_bytes(b"abcd")
            with self.assertRaisesRegex(CORPUS.CorpusValidationError, "prompt hash mismatch"):
                CORPUS.load_corpus_manifest(path)

    def test_manifest_rejects_traversal_duplicates_and_bad_digest(self):
        manifest = CORPUS.load_corpus_manifest(MANIFEST)
        invalid = copy.deepcopy(manifest)
        invalid["prompts"][0]["file"] = "../escape.txt"
        with self.assertRaisesRegex(CORPUS.CorpusValidationError, "portable relative path"):
            CORPUS.validate_corpus_manifest(invalid)

        invalid = copy.deepcopy(manifest)
        invalid["prompts"][1]["id"] = invalid["prompts"][0]["id"]
        with self.assertRaisesRegex(CORPUS.CorpusValidationError, "duplicate prompt id"):
            CORPUS.validate_corpus_manifest(invalid)

        invalid = copy.deepcopy(manifest)
        invalid["tokenizer"]["sha256"] = "ABC"
        with self.assertRaisesRegex(CORPUS.CorpusValidationError, "tokenizer.sha256"):
            CORPUS.validate_corpus_manifest(invalid)


if __name__ == "__main__":
    unittest.main()
