import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("run.py")
SPEC = importlib.util.spec_from_file_location("wackmall_bench_run", MODULE_PATH)
RUN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN)


class PreservedSidecarTests(unittest.TestCase):
    def test_cold_run_captures_generated_sidecar_and_restores_original(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model = root / "model.gguf"
            sidecar = Path(str(model) + ".tier")
            capture = root / "capture" / "result.tier"
            sidecar.write_bytes(b"original")

            with RUN.preserved_sidecar(model, "cold", capture):
                self.assertFalse(sidecar.exists())
                sidecar.write_bytes(b"generated")

            self.assertEqual(sidecar.read_bytes(), b"original")
            self.assertEqual(capture.read_bytes(), b"generated")
            self.assertEqual(list(root.glob("*.bench-backup-*")), [])

    def test_warm_run_restores_original_after_mutation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model = root / "model.gguf"
            sidecar = Path(str(model) + ".tier")
            capture = root / "result.tier"
            sidecar.write_bytes(b"warm-start")

            with RUN.preserved_sidecar(model, "warm", capture):
                self.assertEqual(sidecar.read_bytes(), b"warm-start")
                sidecar.write_bytes(b"updated")

            self.assertEqual(sidecar.read_bytes(), b"warm-start")
            self.assertEqual(capture.read_bytes(), b"updated")

    def test_exception_still_restores_original(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model = root / "model.gguf"
            sidecar = Path(str(model) + ".tier")
            capture = root / "result.tier"
            sidecar.write_bytes(b"original")

            with self.assertRaisesRegex(RuntimeError, "boom"):
                with RUN.preserved_sidecar(model, "cold", capture):
                    sidecar.write_bytes(b"partial")
                    raise RuntimeError("boom")

            self.assertEqual(sidecar.read_bytes(), b"original")

    def test_no_original_leaves_no_sidecar_after_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model = root / "model.gguf"
            sidecar = Path(str(model) + ".tier")
            capture = root / "result.tier"

            with RUN.preserved_sidecar(model, "warm", capture):
                sidecar.write_bytes(b"new")

            self.assertFalse(sidecar.exists())
            self.assertEqual(capture.read_bytes(), b"new")


class NormalizationTests(unittest.TestCase):
    def test_parse_json_accepts_dict_and_filters_list(self):
        self.assertEqual(RUN.parse_llama_bench_json('{"n_prompt": 4}'), [{"n_prompt": 4}])
        self.assertEqual(
            RUN.parse_llama_bench_json('[{"n_gen": 2}, 7, null]'),
            [{"n_gen": 2}],
        )
        self.assertEqual(RUN.parse_llama_bench_json("not-json"), [])

    def test_throughput_rows_classifies_modes(self):
        rows = RUN.throughput_rows([
            {"n_prompt": 16, "n_gen": 0, "avg_ts": 10.0},
            {"n_prompt": 0, "n_gen": 8, "avg_ts": 20.0},
            {"n_prompt": 4, "n_gen": 2, "avg_ts": 30.0},
        ])
        self.assertEqual([row["mode"] for row in rows], ["PP", "TG", "PG"])

    def test_csv_rows_preserve_normalized_measurement_fields(self):
        results = [{
            "run_id": "0001-pp-16-s2-tmax3-cold",
            "mode": "PP",
            "tokens": 16,
            "batch": 4,
            "ubatch": 2,
            "threads": 1,
            "s": 2,
            "tmax": 3,
            "sidecar": "cold",
            "returncode": 0,
            "wall_s": 1.25,
            "throughput": [{
                "mode": "PP",
                "n_prompt": 16,
                "n_gen": 0,
                "n_depth": 0,
                "avg_ts": 123.5,
                "stddev_ts": 1.5,
                "n_batch": 4,
                "n_ubatch": 2,
                "n_threads": 1,
                "backend": "CUDA",
            }],
        }]

        row = RUN.csv_result_rows(results)[0]
        self.assertEqual(row["bench_mode"], "PP")
        self.assertEqual(row["n_prompt"], 16)
        self.assertEqual(row["n_gen"], 0)
        self.assertEqual(row["n_depth"], 0)
        self.assertEqual(row["bench_n_batch"], 4)
        self.assertEqual(row["bench_n_ubatch"], 2)
        self.assertEqual(row["bench_n_threads"], 1)
        self.assertEqual(row["avg_ts"], 123.5)
        self.assertEqual(row["backend"], "CUDA")


if __name__ == "__main__":
    unittest.main()
