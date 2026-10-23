"""Release invariants for complete, source-attributed experiment documentation."""
import hashlib
import json
from pathlib import Path
import unittest

from hastika.common.paths import PROJECT_ROOT


class ExperimentCatalogueContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = json.loads((PROJECT_ROOT / "results/experiment_catalog.json").read_text())

    def test_every_original_has_a_notebook_and_individual_result(self):
        entries = self.catalog["experiments"]
        self.assertEqual(len(entries), 81)
        self.assertEqual(len({r["notebook"] for r in entries}), 81)
        self.assertEqual(sum(r["task"] == "a" for r in entries), 39)
        self.assertEqual(sum(r["task"] == "b" for r in entries), 42)
        for entry in entries:
            with self.subTest(notebook=entry["notebook"]):
                notebook = json.loads((PROJECT_ROOT / entry["notebook"]).read_text())
                record = json.loads((PROJECT_ROOT / entry["result_record"]).read_text())
                self.assertEqual(notebook["metadata"]["project_manas"]["result_record"], entry["result_record"])
                self.assertEqual(record["outcome"], entry["outcome"])
                self.assertTrue(record["source"]["notebook_sha256"])
                self.assertTrue(record["status"])
                self.assertEqual(record["individual_official_final_score"], None)

    def test_historical_code_identity_outputs_and_exclusion_boundary(self):
        counts = {"historical_snapshot": 0, "exclusion_record": 0}
        for entry in self.catalog["experiments"]:
            notebook = json.loads((PROJECT_ROOT / entry["notebook"]).read_text())
            code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
            counts[entry["notebook_kind"]] += 1
            if entry["notebook_kind"] == "exclusion_record":
                self.assertFalse(code_cells)
                self.assertTrue(json.loads((PROJECT_ROOT / entry["result_record"]).read_text())["exclusion_reason"])
                continue
            code = "\n".join("".join(c.get("source", [])) for c in code_cells)
            digest = hashlib.sha256(code.encode()).hexdigest()
            self.assertEqual(digest, notebook["metadata"]["project_manas"]["source_code_cells_sha256"])
            for token in ["hastika.task_a.leak", "--transductive", "--allow-transductive", "hate_only_compiled_multiclass"]:
                self.assertNotIn(token, code, entry["notebook"])
            for cell in code_cells:
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])
        self.assertEqual(counts, {"historical_snapshot": 72, "exclusion_record": 9})

    def test_metric_extracts_are_bounded_and_source_attributed(self):
        for entry in self.catalog["experiments"]:
            record = json.loads((PROJECT_ROOT / entry["result_record"]).read_text())
            for log in record["logs"]:
                self.assertEqual(len(log["sha256"]), 64)
                for metric in log["metric_summaries"]:
                    self.assertGreaterEqual(metric["macro_f1"], 0)
                    self.assertLessEqual(metric["macro_f1"], 1)
                    self.assertGreater(metric["line"], 0)
                    if metric["accuracy"] is not None:
                        self.assertGreaterEqual(metric["accuracy"], 0)
                        self.assertLessEqual(metric["accuracy"], 1)
                    for score in metric["per_class_f1"].values():
                        self.assertGreaterEqual(score, 0)
                        self.assertLessEqual(score, 1)
                for comparison in log["paired_comparisons"]:
                    self.assertGreaterEqual(comparison["p_better"], 0)
                    self.assertLessEqual(comparison["p_better"], 1)
                    self.assertLessEqual(comparison["ci95"][0], comparison["ci95"][1])
