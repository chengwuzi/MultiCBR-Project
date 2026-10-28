"""Only seed planning, averaging and a two-run tiny CLI chain; no full training."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from MultiCBR import run_repeated as runner
from MultiCBR.test_reproduction import fixture


class RepeatedTests(unittest.TestCase):
    def test_random_seed_plan_saved_before_training_and_invalid_requests_rejected(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            path, _ = fixture(root, "Youshu")
            command = ["-d", "Youshu", "-n", "5", "--config", str(path), "--dry-run",
                       "--output-dir", str(root / "batches"), "--record-file", str(root / "record.txt")]
            with patch.object(runner, "execute_run", side_effect=AssertionError("Preview must not train")):
                self.assertEqual(runner.main(command), 0)
            directory = next((root / "batches/Youshu").iterdir())
            batch = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            seeds = [run["seed"] for run in batch["runs"]]
            self.assertEqual(len(seeds), 5)
            self.assertEqual(len(set(seeds)), 5)
            self.assertTrue(all(0 <= seed < 2**32 for seed in seeds))
            self.assertEqual(batch["seed_origin"], "random_without_replacement")
            self.assertTrue(all(run["status"] == "PENDING" for run in batch["runs"]))
            self.assertFalse((root / "record.txt").exists())
            runner.verify_inputs(batch)
            summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
            self.assertFalse(summary["complete"])
            self.assertIsNone(summary["mean"])
            for args in (["-d", "Youshu", "-n", "0"], ["-d", "Youshu", "-n", "2", "--seeds", "1", "1"]):
                with self.assertRaises(ValueError):
                    runner.main(args)
            batch["runs"][0]["seed"] = (seeds[0] + 1) % 2**32
            with self.assertRaisesRegex(ValueError, "changed"):
                runner.verify_inputs(batch)

    def test_two_real_subprocess_runs_mean_std_resume_and_incomplete_summary(self):
        import statistics
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            path, _ = fixture(root, "NetEase")
            args = ["-d", "NetEase", "-n", "2", "--seeds", "17", "53", "--config", str(path),
                    "--device", "cpu", "--output-dir", str(root / "batches"),
                    "--record-file", str(root / "record.txt")]
            self.assertEqual(runner.main(args), 0)
            directory = next((root / "batches/NetEase").iterdir())
            batch = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["successful"], 2)
            self.assertEqual(summary["seeds"], [17,53])
            self.assertNotEqual(batch["runs"][0]["execution_id"], batch["runs"][1]["execution_id"])
            for split in ("val", "test"):
                for metric in ("recall", "ndcg"):
                    for k in ("10", "20", "40", "80"):
                        values = [item["metrics"][split][metric][k] for item in summary["runs"]]
                        self.assertEqual(summary["mean"][split][metric][k], statistics.mean(values))
                        self.assertEqual(summary["std"][split][metric][k], statistics.stdev(values))
            before = (root / "record.txt").read_bytes()
            with patch.object(runner, "execute_run", side_effect=AssertionError("Successful runs must not repeat")):
                self.assertEqual(runner.main(["--resume", str(directory / "manifest.json")]), 0)
            self.assertEqual((root / "record.txt").read_bytes(), before)
            self.assertEqual(before.count(b"END_EXECUTION:"), 2)
            # An incomplete planned run must prevent a partial formal mean.
            batch["runs"][1]["status"] = "FAILED"
            partial = runner.summarize(batch, directory)
            self.assertFalse(partial["complete"])
            self.assertEqual(partial["successful"], 1)
            self.assertIsNone(partial["mean"])
            self.assertIsNone(partial["std"])
            # Changed full-precision selection evidence cannot enter the mean.
            result = Path(batch["runs"][0]["attempts"][-1]["result_json"])
            record = json.loads(result.read_text(encoding="utf-8"))
            record["result"]["selection_metrics"]["recall@20"] = .999
            result.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Selection"):
                runner.validate_run(batch, batch["runs"][0])


if __name__ == "__main__":
    unittest.main()
