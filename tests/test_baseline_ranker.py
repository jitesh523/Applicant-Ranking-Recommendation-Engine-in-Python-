import unittest

import numpy as np
import pandas as pd

from baseline_ranker import build_features


class BaselineFeatureTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({
            "Experience (Years)": [0, 1, 2, 3],
            "Salary Expectation ($)": [50000] * 4,
            "Projects Count": [2] * 4,
            "Education": ["Degree A", "Degree B", "Degree A", "Degree B"],
            "Certifications": [None, "Cert", None, "Cert"],
            "AI Score (0-100)": [20, 40, 60, 80],
            "Job Role": ["Synthetic A", "Synthetic A", "Synthetic B", "Synthetic B"],
        })

    def test_feature_matrix_is_dense_and_preserves_row_alignment(self):
        frame = self.frame()
        features, labels, groups, roles, encoder, numeric, categorical = build_features(frame)
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(features.shape, (4, 7))
        np.testing.assert_array_equal(features[:, :3], frame[numeric].to_numpy())
        np.testing.assert_array_equal(labels, frame["AI Score (0-100)"].to_numpy())
        np.testing.assert_array_equal(roles, frame["Job Role"].to_numpy())
        np.testing.assert_array_equal(groups, [2, 2])
        unseen = pd.DataFrame({"Education": ["Unseen"], "Certifications": ["Unseen"]})
        np.testing.assert_array_equal(encoder.transform(unseen[categorical]), np.zeros((1, 4)))

    def test_missing_categories_still_produce_finite_dense_features(self):
        frame = self.frame()
        frame[["Education", "Certifications"]] = None
        features, _, _, _, encoder, _, _ = build_features(frame)
        self.assertEqual(features.shape, (4, 5))
        self.assertTrue(np.isfinite(features).all())
        for categories in encoder.categories_:
            np.testing.assert_array_equal(categories, ["Unknown"])


if __name__ == "__main__":
    unittest.main()
