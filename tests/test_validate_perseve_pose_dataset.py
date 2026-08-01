"""Focused tests for the standalone pose dataset validation report."""

import hashlib
import json
import unittest

from validate_perseve_pose_dataset import summarize_checksum_set


class PersevePoseDatasetValidatorTests(unittest.TestCase):
    def test_checksum_set_summary_is_unique_compact_and_deterministic(self):
        forward = summarize_checksum_set(["b" * 64, "a" * 64, "b" * 64])
        reverse = summarize_checksum_set(["a" * 64, "b" * 64])
        expected_payload = json.dumps(
            ["a" * 64, "b" * 64],
            separators=(",", ":"),
        ).encode("utf-8")

        self.assertEqual(forward, reverse)
        self.assertEqual(forward["count"], 2)
        self.assertEqual(
            forward["digest_sha256"],
            hashlib.sha256(expected_payload).hexdigest(),
        )
        self.assertNotIn("values", forward)


if __name__ == "__main__":
    unittest.main()
