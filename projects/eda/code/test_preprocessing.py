"""Testes rápidos do pré-processamento, sem treinar classificador."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from preprocessing import MODEL_FEATURES, SKEWED_NUMERIC, build_preprocessor


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        # Todos os atributos necessários; o teste nunca ajusta o pipeline nos dados novos.
        self.train = pd.DataFrame(
            {
                "Administrative": [0, 1, 2, 3],
                "Administrative_Duration": [0.0, 1.0, 2.0, 3.0],
                "Informational": [0, 1, 2, 3],
                "Informational_Duration": [0.0, 1.0, 2.0, 3.0],
                "ProductRelated": [1, 2, 3, 4],
                "ProductRelated_Duration": [1.0, 2.0, 3.0, 4.0],
                "BounceRates": [0.0, 0.1, 0.0, 0.1],
                "ExitRates": [0.0, 0.1, 0.0, 0.1],
                "SpecialDay": [0.0, 0.0, 0.5, 1.0],
                "Month": ["Feb", "Mar", "Feb", "Mar"],
                "OperatingSystems": [1, 2, 1, 2],
                "Browser": [1, 2, 1, 2],
                "Region": [1, 2, 1, 2],
                "TrafficType": [1, 2, 1, 2],
                "VisitorType": ["New_Visitor", "Returning_Visitor"] * 2,
                "Weekend": [False, True, False, True],
            }
        )[MODEL_FEATURES]

    def test_unknown_category_and_missing_values(self):
        preprocessor = build_preprocessor()
        transformed_train = preprocessor.fit_transform(self.train)
        new_row = self.train.iloc[[0]].copy()
        new_row.loc[:, "Month"] = "UnseenMonth"
        new_row["Administrative"] = new_row["Administrative"].astype(float)
        new_row.loc[:, "Administrative"] = np.nan
        transformed_new = preprocessor.transform(new_row)
        self.assertEqual(transformed_train.shape[1], transformed_new.shape[1])
        self.assertTrue(np.isfinite(transformed_new).all())
        self.assertEqual(len(preprocessor.get_feature_names_out()), transformed_train.shape[1])
        month_names = np.array(preprocessor.get_feature_names_out())
        month_columns = np.flatnonzero(np.char.startswith(month_names.astype(str), "cat__Month_"))
        self.assertTrue((transformed_new[0, month_columns] == 0).all())

    def test_outlier_limit_is_learned_only_from_train(self):
        preprocessor = build_preprocessor()
        preprocessor.fit(self.train)
        caps_before = preprocessor.named_transformers_["skew"].named_steps["cap"].upper_.copy()
        new_row = self.train.iloc[[0]].copy()
        new_row.loc[:, SKEWED_NUMERIC] = 1_000_000
        self.assertTrue(np.isfinite(preprocessor.transform(new_row)).all())
        caps_after = preprocessor.named_transformers_["skew"].named_steps["cap"].upper_
        np.testing.assert_array_equal(caps_before, caps_after)


if __name__ == "__main__":
    unittest.main()
