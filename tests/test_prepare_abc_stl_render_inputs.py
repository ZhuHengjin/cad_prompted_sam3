"""Tests for recursive ABC STL render-input staging and validation."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from prepare_abc_stl_render_inputs import (
    PreparationError,
    RenderValidationError,
    main,
    prepare_symlink_directory,
    validate_render_directory,
)


class PrepareAbcStlRenderInputsTests(unittest.TestCase):
    def test_recursively_stages_entire_corpus_and_resumes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "abc"
            first = source / "0000" / "00000001" / "alpha.stl"
            second = source / "0001" / "nested" / "beta.STL"
            ignored = source / "0002" / "notes.txt"
            for path in (first, second, ignored):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(path.name, encoding="utf-8")

            staging = root / "flat"
            meshes, first_summary = prepare_symlink_directory(source, staging)

            self.assertEqual(meshes, [first, second])
            self.assertEqual(first_summary["stl_count"], 2)
            self.assertEqual(first_summary["symlinks"], {"created": 2, "existing": 0, "total": 2})
            self.assertEqual({path.name for path in staging.iterdir()}, {"alpha.stl", "beta.STL"})
            self.assertTrue((staging / "alpha.stl").is_symlink())
            self.assertEqual((staging / "alpha.stl").resolve(), first.resolve())
            self.assertEqual((staging / "beta.STL").resolve(), second.resolve())

            _, second_summary = prepare_symlink_directory(source, staging)
            self.assertEqual(second_summary["symlinks"], {"created": 0, "existing": 2, "total": 2})

    def test_duplicate_stems_are_rejected_before_creating_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "abc"
            lower = source / "one" / "part.stl"
            upper = source / "two" / "PART.STL"
            for path in (lower, upper):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"solid")
            staging = root / "flat"

            with self.assertRaisesRegex(PreparationError, "unique.*case-insensitive"):
                prepare_symlink_directory(source, staging)

            self.assertFalse(staging.exists())

    def test_output_inside_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "abc"
            source.mkdir()
            (source / "part.stl").write_bytes(b"solid")

            with self.assertRaisesRegex(PreparationError, "must not.*inside"):
                prepare_symlink_directory(source, source / "flat")

            self.assertFalse((source / "flat").exists())

            external = root / "external"
            external.mkdir()
            linked_output = source / "linked-flat"
            linked_output.symlink_to(external, target_is_directory=True)
            with self.assertRaisesRegex(PreparationError, "must not.*inside"):
                prepare_symlink_directory(source, linked_output)

            self.assertEqual(list(external.iterdir()), [])

    def test_broken_staging_directory_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "abc"
            source.mkdir()
            (source / "part.stl").write_bytes(b"solid")
            staging = root / "flat"
            missing_target = root / "missing"
            staging.symlink_to(missing_target, target_is_directory=True)

            with self.assertRaisesRegex(PreparationError, "broken symlink"):
                prepare_symlink_directory(source, staging)

            self.assertTrue(staging.is_symlink())
            self.assertFalse(missing_target.exists())

    def test_regular_wrong_and_stale_entries_are_never_overwritten(self):
        scenarios = ("regular", "wrong_symlink", "stale_symlink", "extra")
        for scenario in scenarios:
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                source = root / "abc"
                source.mkdir()
                mesh = source / "part.stl"
                mesh.write_bytes(b"solid")
                staging = root / "flat"
                staging.mkdir()
                expected_link = staging / mesh.name
                if scenario == "regular":
                    expected_link.write_text("keep me", encoding="utf-8")
                elif scenario == "wrong_symlink":
                    other = root / "other.stl"
                    other.write_bytes(b"other")
                    expected_link.symlink_to(other)
                elif scenario == "stale_symlink":
                    expected_link.symlink_to(root / "missing.stl")
                else:
                    (staging / "unexpected.stl").symlink_to(mesh)

                with self.assertRaisesRegex(PreparationError, "nothing was changed"):
                    prepare_symlink_directory(source, staging)

                if scenario == "regular":
                    self.assertEqual(expected_link.read_text(encoding="utf-8"), "keep me")
                elif scenario == "wrong_symlink":
                    self.assertEqual(expected_link.resolve().name, "other.stl")
                elif scenario == "stale_symlink":
                    self.assertTrue(expected_link.is_symlink())
                    self.assertFalse(expected_link.exists())
                else:
                    self.assertFalse(expected_link.exists())

    def test_render_validation_accepts_padded_and_unpadded_complete_pairs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mesh = root / "part.stl"
            mesh.write_bytes(b"solid")
            render_dir = root / "renders"
            render_dir.mkdir()
            for view_id in range(12):
                label = f"{view_id:02d}" if view_id % 2 == 0 else str(view_id)
                base = render_dir / f"part_stl_base_{label}"
                base.with_suffix(".png").write_bytes(b"image")
                base.with_name(f"{base.name}_mask.png").write_bytes(b"mask")

            report = validate_render_directory([mesh], render_dir)

            self.assertEqual(report["expected_pair_count"], 12)
            self.assertEqual(report["complete_pair_count"], 12)
            self.assertEqual(report["incomplete_pair_count"], 0)

    def test_render_validation_requires_both_files_for_every_view(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mesh = root / "part.stl"
            mesh.write_bytes(b"solid")
            render_dir = root / "renders"
            render_dir.mkdir()
            for view_id in range(12):
                base = render_dir / f"part_stl_base_{view_id:02d}"
                base.with_suffix(".png").write_bytes(b"image")
                if view_id != 7:
                    base.with_name(f"{base.name}_mask.png").write_bytes(b"mask")

            with self.assertRaises(RenderValidationError) as caught:
                validate_render_directory([mesh], render_dir)

            report = caught.exception.report
            self.assertEqual(report["complete_pair_count"], 11)
            self.assertEqual(report["incomplete_pair_count"], 1)
            self.assertEqual(report["incomplete_object_count"], 1)
            self.assertEqual(report["issue_examples"][0]["view_id"], 7)
            padded = report["issue_examples"][0]["candidates"][0]
            self.assertTrue(padded["image_exists"])
            self.assertFalse(padded["mask_exists"])

    def test_cli_prints_json_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "abc"
            source.mkdir()
            (source / "part.stl").write_bytes(b"solid")
            output = io.StringIO()

            with contextlib.redirect_stdout(output):
                exit_code = main([str(source), str(root / "flat")])

            self.assertEqual(exit_code, 0)
            summary = json.loads(output.getvalue())
            self.assertEqual(summary["status"], "ok")
            self.assertEqual(summary["stl_count"], 1)
            self.assertIsNone(summary["render_validation"])


if __name__ == "__main__":
    unittest.main()
