import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import lightgbm as lgb

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


class RankingLabelTests(unittest.TestCase):
    def frame(self, scores):
        n = len(scores)
        return pd.DataFrame({
            "Job Role": ["Synthetic Role"] * n,
            "Skills": ["python,sql"] * n,
            "Experience (Years)": np.arange(n),
            "Salary Expectation ($)": [50000] * n,
            "Projects Count": [2] * n,
            "Education": ["B.Sc"] * n,
            "Certifications": [None] * n,
            "AI Score (0-100)": scores,
        })

    def test_relevance_labels_include_zero_and_bin_boundaries(self):
        scores = [0, 20, 20.01, 40, 40.01, 60, 60.01, 80, 80.01, 100]
        with patch("pipeline.lgb.LGBMRanker") as ranker:
            RankingPipeline().fit(self.frame(scores))
        labels = ranker.return_value.fit.call_args.args[1]
        np.testing.assert_array_equal(labels, [0, 0, 1, 1, 2, 2, 3, 3, 4, 4])

    def test_zero_score_is_accepted_by_native_lightgbm(self):
        native = lgb.LGBMRanker(objective="lambdarank", n_estimators=2,
                               min_child_samples=1, n_jobs=1, verbosity=-1)
        with patch("pipeline.lgb.LGBMRanker", return_value=native):
            pipeline = RankingPipeline()
            pipeline.fit(self.frame([0, 25, 75, 100]))
        self.assertIs(pipeline.model, native)
        self.assertTrue(hasattr(native, "booster_"))

    def test_invalid_scores_are_rejected_before_model_training(self):
        for score in [-1, 101, np.nan, np.inf]:
            with self.subTest(score=score), patch("pipeline.lgb.LGBMRanker") as ranker:
                with self.assertRaisesRegex(ValueError, "AI Score"):
                    RankingPipeline().fit(self.frame([score, 50]))
                ranker.assert_not_called()


if __name__ == "__main__":
    unittest.main()
