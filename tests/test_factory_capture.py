import os
import unittest
from unittest.mock import patch

from sim.factory.capture_filters import (
    VALID_BINS,
    VALID_SPLITS,
    filter_records,
    selection,
)


class CaptureSelectionTests(unittest.TestCase):
    def test_empty_filter_means_all_values(self):
        with patch.dict(os.environ, {"CHEESE_CAPTURE_SPLITS": ""}):
            self.assertIsNone(selection("CHEESE_CAPTURE_SPLITS", VALID_SPLITS))

    def test_filter_is_trimmed_and_deduplicated(self):
        with patch.dict(
            os.environ,
            {"CHEESE_CAPTURE_BINS": " bin_blue,bin_fresh,bin_blue "},
        ):
            self.assertEqual(
                selection("CHEESE_CAPTURE_BINS", VALID_BINS),
                {"bin_blue", "bin_fresh"},
            )

    def test_unknown_filter_value_fails_closed(self):
        with patch.dict(os.environ, {"CHEESE_CAPTURE_SPLITS": "training"}):
            with self.assertRaisesRegex(ValueError, "unsupported values"):
                selection("CHEESE_CAPTURE_SPLITS", VALID_SPLITS)

    def test_record_filter_combines_split_and_bin(self):
        records = [
            {"uid": "keep", "split": "train", "bin": "bin_blue"},
            {"uid": "wrong-split", "split": "val", "bin": "bin_blue"},
            {"uid": "wrong-bin", "split": "train", "bin": "bin_hard"},
        ]
        self.assertEqual(
            filter_records(records, {"train"}, {"bin_blue"}),
            [records[0]],
        )


if __name__ == "__main__":
    unittest.main()
