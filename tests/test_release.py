import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import pandas as pd

from hastika.common.paths import PROJECT_ROOT
from hastika.common.preprocessing import clean, dedupe_index
from hastika.common.tasks import TASKS, FILES
from hastika.common.validation import read_frame, check_disjoint
from hastika.data.external import prepare_external
from hastika.data.splits import prepare, fingerprint
from hastika.evaluation.ensemble import write_bundle, read_bundle, combine, package
from hastika.evaluation.metrics import evaluate
from hastika.runner import load_recipe, commands, execute, assemble


class ReleaseContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def bundle(self, name, probs=None, ids=("001", "002"), members=1):
        path = self.root / name
        write_bundle(path, probs if probs is not None else [[.8, .2], [.1, .9]], ids, "a", members)
        return path

    def test_cleaning_and_conflicting_duplicates(self):
        self.assertEqual(clean("x<br>y &amp; z"), "x y & z")
        self.assertEqual(dedupe_index(["same", "SAME", "ambiguous", "ambiguous", "other"],
                                      ["Hate", "Hate", "Hate", "Non-Hate", "Hate"]), [0, 4])

    def test_sources_reject_wrong_task_labels_and_overlap(self):
        path = self.root / "wrong.csv"
        pd.DataFrame({"id": ["1"], "Comment": ["synthetic example"], "Label": ["Gender"]}).to_csv(path, index=False)
        with self.assertRaises(ValueError):
            read_frame(path, "a", labelled=True)
        a = pd.DataFrame({"id": ["1"], "Comment": ["same text"]})
        b = pd.DataFrame({"id": ["2"], "Comment": ["SAME text"]})
        with self.assertRaises(ValueError):
            check_disjoint(a, b)
        self.assertEqual(check_disjoint(a, b, check_text=False)["ids"], 0)

    def test_synthetic_splits_are_reproducible_and_disjoint(self):
        spec = FILES["a"]
        rows = [{"id": str(i), "Comment": f"synthetic unique example number {i}",
                 "Label": "Hate" if i % 2 else "Non-Hate"} for i in range(40)]
        pd.DataFrame(rows[:30]).to_csv(self.root / spec["train"], index=False)
        pd.DataFrame(rows[30:]).to_csv(self.root / spec["validation"], index=False)
        first = prepare("a", self.root, self.root / "first", verify=False)
        second = prepare("a", self.root, self.root / "second", verify=False)
        self.assertEqual(first["holdout_fingerprint"], second["holdout_fingerprint"])
        self.assertEqual(first["labelled_rows"], 40)
        self.assertEqual(first["holdout_rows"], 6)
        tr = read_frame(self.root / "first/train.csv", "a", True)
        ho = read_frame(self.root / "first/holdout.csv", "a", True)
        check_disjoint(tr, ho)
        self.assertEqual(first["holdout_fingerprint"], fingerprint(ho.id))

    def test_external_text_exclusion_does_not_export_labels(self):
        source = self.root / "external.csv"
        pd.DataFrame({"Comment": ["keep text", "held out example"], "Label": ["unused_a", "unused_b"]}).to_csv(source, index=False)
        excluded = pd.DataFrame({"id": ["2"], "Comment": ["HELD out example"]})
        report = prepare_external(source, [excluded], self.root / "filtered.csv")
        result = pd.read_csv(self.root / "filtered.csv")
        self.assertEqual(result.columns.tolist(), ["Comment"])
        self.assertEqual(result.Comment.tolist(), ["keep text"])
        self.assertEqual(report["excluded_rows"], 1)

    def test_probability_validation(self):
        for p in [[[float("nan"), .2], [.1, .9]], [[-.1, 1.1], [.1, .9]], [[.8, .8], [.1, .9]]]:
            with self.assertRaises(ValueError):
                self.bundle("invalid", p)
        with self.assertRaises(ValueError):
            self.bundle("duplicate_ids", ids=("1", "1"))
        with self.assertRaises(ValueError):
            self.bundle("zero_members", members=0)

    def test_weighted_average_and_member_counts(self):
        a = self.bundle("a", [[.9, .1], [.2, .8]], members=4)
        b = self.bundle("b", [[.6, .4], [.3, .7]], members=2)
        out = self.root / "combined"
        combine([a, b], "a", out, expected_members=6)
        p, ids, meta = read_bundle(out, "a")
        np.testing.assert_allclose(p, (np.array([[.9, .1], [.2, .8]]) * 4 + np.array([[.6, .4], [.3, .7]]) * 2) / 6)
        self.assertEqual(ids, ["001", "002"])
        self.assertEqual(meta["member_count"], 6)
        with self.assertRaises(ValueError):
            combine([a], "a", out, expected_members=6)

    def test_shuffled_ids_wrong_task_and_column_order_rejected(self):
        a = self.bundle("a")
        b = self.bundle("b", ids=("002", "001"))
        with self.assertRaises(ValueError):
            combine([a, b], "a", self.root / "bad")
        with self.assertRaises(ValueError):
            read_bundle(a, "b")
        meta = json.loads((self.root / "a_metadata.json").read_text())
        meta["classes"].reverse()
        (self.root / "a_metadata.json").write_text(json.dumps(meta))
        with self.assertRaises(ValueError):
            read_bundle(a, "a")

    def test_collapsed_members_are_rejected(self):
        a = self.bundle("collapsed", [[.8, .2], [.7, .3]])
        with self.assertRaises(ValueError):
            combine([a], "a", self.root / "bad")

    def test_package_and_metrics_require_reference_alignment(self):
        a = self.bundle("a")
        ref = self.root / "ref.csv"
        pd.DataFrame({"id": ["001", "002"], "Comment": ["example one", "example two"],
                      "Label": ["Non-Hate", "Hate"]}).to_csv(ref, index=False)
        package(a, ref, self.root / "submission.zip")
        with zipfile.ZipFile(self.root / "submission.zip") as z:
            self.assertEqual(z.namelist(), ["predictions.csv"])
            self.assertTrue(z.read("predictions.csv").startswith(b"id,label\n"))
        self.assertEqual(evaluate("a", ref, a)["macro_f1"], 1.0)
        frame = pd.read_csv(ref, dtype=str).iloc[::-1]
        frame.to_csv(ref, index=False)
        with self.assertRaises(ValueError):
            package(a, ref, self.root / "bad.zip")
        with self.assertRaises(ValueError):
            evaluate("a", ref, a)

    def test_final_configs_and_dry_run_job_composition(self):
        official = json.loads((PROJECT_ROOT / "results/final_scores.json").read_text())
        self.assertEqual(official["a"]["macro_f1"], .825049)
        self.assertEqual(official["b"]["macro_f1"], .736678008)
        for task, expected, jobs_count in [("a", 12, 17), ("b", 8, 12)]:
            cfg = load_recipe(PROJECT_ROOT / f"configs/task_{task}/submitted.json")
            plan = commands(cfg, self.root / "dry", self.root / "raw")
            self.assertEqual(len(plan), jobs_count)
            self.assertEqual(sum(j["stage"] == "classifier" for j in plan), expected)
            self.assertEqual(cfg["member_count"], expected)
            self.assertFalse((self.root / "dry").exists())

    def test_runner_complete_fake_training_and_recorded_collapse_retry(self):
        # Exercise orchestration/assembly without downloading or training a model.
        task = "a"
        cfg = load_recipe(PROJECT_ROOT / "configs/task_a/submitted.json")
        raw = self.root / "raw"
        raw.mkdir()
        pd.DataFrame({"id": ["001", "002"], "Comment": ["test example one", "test example two"]}).to_csv(raw / FILES[task]["test"], index=False)
        external = self.root / "external.csv"
        pd.DataFrame({"Comment": ["external example"]}).to_csv(external, index=False)
        out = self.root / "run"
        expected_external_hash = json.loads((PROJECT_ROOT / "data/sources.json").read_text())["external"]["sha256"]
        def fake_prepare(task, raw, data):
            data.mkdir(parents=True)
            pd.DataFrame({"id": ["tr1", "tr2"], "Comment": ["train one", "train two"], "Label": ["Hate", "Non-Hate"]}).to_csv(data / "all.csv", index=False)
            return {"fixture": True}
        retried = set()
        def fake_launch(cmd, log):
            destination = Path(cmd[cmd.index("--out") + 1])
            if "hastika.models.llm_tapt" in cmd:
                destination.mkdir(parents=True)
                return 0
            seed = int(cmd[cmd.index("--seed") + 1])
            if seed == 42 and destination.parent.name == "run33" and seed not in retried:
                retried.add(seed)
                return 4
            prediction = Path(cmd[cmd.index("--predict") + 1])
            write_bundle(destination / prediction.stem, [[.8, .2], [.1, .9]], ["001", "002"], task)
            return 0
        with patch("hastika.runner.prepare", side_effect=fake_prepare), patch("hastika.runner.sha256", return_value=expected_external_hash), patch("hastika.runner.launch", side_effect=fake_launch):
            execute(cfg, out, raw, external, None)
        _, _, meta = read_bundle(out / "final", "a")
        self.assertEqual(meta["member_count"], 12)
        events = json.loads((out / "run33/execution.json").read_text())
        self.assertEqual(events[1]["effective_seed"], 1042)
        broken = dict(cfg, mode="holdout")
        with self.assertRaises(ValueError):
            assemble(broken, out)


if __name__ == "__main__":
    unittest.main()
