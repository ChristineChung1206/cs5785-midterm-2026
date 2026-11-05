"""Export contracts for the portable layout; no estimator is fitted."""
import _bootstrap  # noqa: F401

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import reproduce


def registry_fixture(root):
    selection, payloads = {}, {
        "part1": b"ID,Predicted\nA,Safety\n",
        "part2": b"ID,label\nB,0.12345678901234567\n",
    }
    for part, content in payloads.items():
        relative = f"models/{part}/final/submission.csv"
        source = root / relative
        source.parent.mkdir(parents=True)
        source.write_bytes(content)
        selection[part] = {"best_public": {"submission": relative,
            "sha256": hashlib.sha256(content).hexdigest()}}
    (root / "models/selection.json").write_text(json.dumps(selection))
    return selection, payloads


class SubmissionExportTests(unittest.TestCase):
    def test_export_preserves_canonical_bytes_without_external_directories(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "midterm"
            selection, payloads = registry_fixture(root)
            output = Path(temporary) / "new_output"
            result = reproduce.export_best_submissions(root, output)
            self.assertEqual(sorted(p.name for p in output.iterdir()), ["part1.csv", "part2.csv"])
            for part, contents in payloads.items():
                self.assertEqual((output / f"{part}.csv").read_bytes(), contents)
                self.assertEqual(result[part]["sha256"], selection[part]["best_public"]["sha256"])
                self.assertEqual((root / selection[part]["best_public"]["submission"]).read_bytes(), contents)
            self.assertFalse((root / "tools/archive").exists())
            self.assertFalse((root.parent / "deliverables").exists())

    def test_empty_directory_is_allowed_but_nonempty_is_rejected_before_inference(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "midterm"
            registry_fixture(root)
            output = Path(temporary) / "output"
            output.mkdir()
            reproduce.export_best_submissions(root, output)
            before = {p.name: p.read_bytes() for p in output.iterdir()}
            with patch.object(reproduce, "check_registry") as check:
                with self.assertRaises(FileExistsError):
                    reproduce.main(["--project-root", str(root), "--output-dir", str(output)])
                check.assert_not_called()
            self.assertEqual(before, {p.name: p.read_bytes() for p in output.iterdir()})

    def test_checksum_failure_creates_no_output(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "midterm"
            selection, _ = registry_fixture(root)
            (root / selection["part2"]["best_public"]["submission"]).write_bytes(b"changed")
            output = Path(temporary) / "output"
            with self.assertRaisesRegex(AssertionError, "bytes changed"):
                reproduce.export_best_submissions(root, output)
            self.assertFalse(output.exists())

    def test_prediction_failure_prevents_export(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"
            with patch.object(reproduce, "check_registry", return_value={}), \
                 patch.object(reproduce, "check_part1", return_value={}), \
                 patch.object(reproduce, "check_part1_best_public", return_value={}), \
                 patch.object(reproduce, "check_part2", side_effect=AssertionError("bad prediction")), \
                 patch.object(reproduce, "export_best_submissions") as export:
                with self.assertRaisesRegex(AssertionError, "bad prediction"):
                    reproduce.main(["--output-dir", str(output)])
                export.assert_not_called()
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
