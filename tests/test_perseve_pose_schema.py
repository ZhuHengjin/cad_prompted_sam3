"""Machine-readable Perseve pose-v1 schema smoke tests."""

import json
import unittest
from pathlib import Path


class PersevePoseSchemaTests(unittest.TestCase):
    def test_bundled_schemas_are_valid_json_and_strict(self):
        root = Path(__file__).resolve().parents[1] / "schemas" / "perseve-pose-v1"
        expected = {
            "dataset-meta.schema.json",
            "objects.schema.json",
            "pose-annotations.schema.json",
        }
        self.assertEqual({path.name for path in root.glob("*.json")}, expected)
        for filename in expected:
            with (root / filename).open("r") as handle:
                schema = json.load(handle)
            self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
            self.assertFalse(schema["additionalProperties"])


if __name__ == "__main__":
    unittest.main()
