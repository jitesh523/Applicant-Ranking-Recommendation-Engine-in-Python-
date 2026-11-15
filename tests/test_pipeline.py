import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from pipeline import RankingPipeline


class RankingGroupTests(unittest.TestCase):
    def assert_training_groups(self, roles):
        # Experience uniquely identifies each row so feature/label alignment can
        # be checked after the pipeline reorders the training data.
        df = pd.DataFrame({
            "Job Role": roles,
            "Skills": ["python,sql"] * len(roles),
            "Experience (Years)": np.arange(len(roles)),
            "Salary Expectation ($)": [50000] * len(roles),
            "Projects Count": [2] * len(roles),
            "Education": ["B.Sc"] * len(roles),
            "Certifications": [None] * len(roles),
            "AI Score (0-100)": [10, 30, 50, 70, 90],
        }, index=[9, 3, 8, 2, 7])
        original = df.copy(deep=True)

        with patch("pipeline.lgb.LGBMRanker") as ranker:
            RankingPipeline().fit(df)

        args, kwargs = ranker.return_value.fit.call_args
        features, labels = args
        row_ids = features[:, 0].astype(int)
        grouped_roles = df["Job Role"].to_numpy()[row_ids]
        np.testing.assert_array_equal(labels, np.arange(len(df))[row_ids])
        self.assertEqual(sorted(row_ids.tolist()), list(range(len(df))))

        offset = 0
        seen = set()
        for size in kwargs["group"]:
            block = grouped_roles[offset:offset + size]
            self.assertEqual(len(set(block)), 1, f"Mixed-role group: {block}")
            self.assertNotIn(block[0], seen)
            seen.add(block[0])
            offset += size
        self.assertEqual(offset, len(df))
        self.assertEqual(seen, set(roles))
        pd.testing.assert_frame_equal(df, original)

    def test_interleaved_roles_are_contiguous_and_aligned(self):
        self.assert_training_groups(["Researcher", "Coder", "Researcher", "Coder", "Researcher"])

    def test_already_grouped_roles_remain_aligned(self):
        self.assert_training_groups(["Researcher", "Researcher", "Researcher", "Coder", "Coder"])


if __name__ == "__main__":
    unittest.main()
