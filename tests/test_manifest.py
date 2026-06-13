import unittest

from meg_decode.manifest import ManifestRecord, parse_h5_filename
from meg_decode.splits import build_split


class ManifestTests(unittest.TestCase):
    def test_parse_h5_filename(self):
        self.assertEqual(
            parse_h5_filename("task_story_math_105923_4.h5"),
            ("task_story_math", "105923", 4),
        )

    def test_intra_split_is_chunk_based(self):
        records = []
        for split, chunks in {"train": range(1, 9), "test": range(9, 11)}.items():
            for task, label in [("rest", 0), ("task_story_math", 1)]:
                for chunk in chunks:
                    records.append(
                        ManifestRecord(
                            path=f"Intra/{split}/{task}_105923_{chunk}.h5",
                            protocol="Intra",
                            split=split,
                            task=task,
                            class_name="rest" if task == "rest" else "math_story",
                            label=label,
                            subject="105923",
                            chunk=chunk,
                        )
                    )
        split = build_split(records, {"protocol": "intra"})
        self.assertTrue(all(record.chunk <= 6 for record in split.train))
        self.assertTrue(all(record.chunk in {7, 8} for record in split.val))
        self.assertTrue(all(record.split == "test" for record in split.test_groups["test"]))


if __name__ == "__main__":
    unittest.main()

