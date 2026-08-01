"""Machine-readable Perseve pose-v1/v2 schema smoke tests."""

import json
import unittest
from pathlib import Path


class PersevePoseSchemaTests(unittest.TestCase):
    def test_bundled_schemas_are_valid_json_and_strict(self):
        expected = {
            "dataset-meta.schema.json",
            "objects.schema.json",
            "pose-annotations.schema.json",
        }
        schemas_root = Path(__file__).resolve().parents[1] / "schemas"
        for version in ("perseve-pose-v1", "perseve-pose-v2"):
            root = schemas_root / version
            self.assertEqual({path.name for path in root.glob("*.json")}, expected)
            for filename in expected:
                with (root / filename).open("r") as handle:
                    schema = json.load(handle)
                self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
                self.assertFalse(schema["additionalProperties"])

    def test_v2_catalog_requires_point_set_not_symmetry(self):
        root = Path(__file__).resolve().parents[1] / "schemas" / "perseve-pose-v2"
        with (root / "objects.schema.json").open("r") as handle:
            schema = json.load(handle)
        required = schema["$defs"]["object"]["required"]
        self.assertIn("point_set", required)
        self.assertNotIn("symmetry", required)


if __name__ == "__main__":
    unittest.main()
