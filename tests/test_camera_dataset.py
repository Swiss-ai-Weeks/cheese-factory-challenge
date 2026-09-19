import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import build_camera_dataset as builder
from src.prepare_camera_sources import eligible_rows, load_review, mask_passes


class CameraSourceTests(unittest.TestCase):
    def test_candidates_exclude_test_and_ambiguous_generic_blue(self):
        rows = [
            {"uid": "good", "source": "cheese_images", "split": "train",
             "label": "stilton", "extra": json.dumps({"variety_name": "Stilton"})},
            {"uid": "test", "source": "cheese_images", "split": "test",
             "label": "stilton", "extra": json.dumps({"variety_name": "Stilton"})},
            {"uid": "ambiguous", "source": "cheese_images", "split": "val",
             "label": "blue", "extra": json.dumps({"variety_name": "blue"})},
        ]
        self.assertEqual([row["uid"] for row in eligible_rows(rows)], ["good"])

    def test_review_rejects_unknown_uid(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "review.csv"
            path.write_text("uid,decision,reason\nunknown,accept,looks good\n")
            with self.assertRaisesRegex(ValueError, "not an eligible"):
                load_review(path, {"known"})

    def test_mask_gate_rejects_tiny_or_fragmented_subject(self):
        good = {
            "fill_ratio": 0.4, "largest_component_ratio": 0.95,
            "center_offset": 0.05, "bbox": (10, 10, 200, 180),
        }
        self.assertTrue(mask_passes(good))
        self.assertFalse(mask_passes({**good, "fill_ratio": 0.02}))
        self.assertFalse(mask_passes({**good, "largest_component_ratio": 0.2}))


class BalancedCameraDatasetTests(unittest.TestCase):
    @staticmethod
    def _row(split, group, view):
        return {
            "uid": f"factory__{group}__v{view}", "split": split,
            "label": "bin_blue", "group": group,
        }

    def test_balancer_requires_three_canonical_views_and_never_uses_test(self):
        rows = []
        for split in ("train", "val", "test"):
            rows.extend(self._row(split, f"{split}-group", view) for view in range(3))
        rows.extend(self._row("train", "incomplete", view) for view in (0, 1))
        with patch.object(builder, "LABELS", ("bin_blue",)), patch.dict(
            builder.GROUP_TARGETS, {"train": 1, "val": 1}, clear=True,
        ):
            selected = builder.select_balanced(rows)
        self.assertEqual(len(selected), 6)
        self.assertEqual({row["split"] for row in selected}, {"train", "val"})
        self.assertNotIn("incomplete", {row["group"] for row in selected})

    def test_view_number_is_fail_closed(self):
        self.assertEqual(builder.view_number("factory__group__v12"), 12)
        self.assertIsNone(builder.view_number("factory__group"))


if __name__ == "__main__":
    unittest.main()
