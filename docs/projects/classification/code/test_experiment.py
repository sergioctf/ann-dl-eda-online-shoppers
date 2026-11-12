"""Verificações rápidas do split, pipeline e MLP; não consulta métricas de teste."""

import unittest

import numpy as np
import torch

from experiment import (
    ShoppingMLP,
    load_inner_split,
    load_outer_split,
    parameter_count,
    transform,
)


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame, cls.train_idx, cls.test_idx, cls.groups = load_outer_split()

    def test_eda_outer_split_matches_documented_counts(self):
        self.assertEqual((len(self.train_idx), len(self.test_idx)), (9864, 2466))
        self.assertEqual(int(self.frame.iloc[self.train_idx]["Revenue"].sum()), 1526)
        self.assertEqual(int(self.frame.iloc[self.test_idx]["Revenue"].sum()), 382)
        self.assertFalse(set(self.groups[self.train_idx]) & set(self.groups[self.test_idx]))

    def test_inner_split_and_pipeline_fit_on_subtrain(self):
        sub_idx, val_idx = load_inner_split(self.frame, self.train_idx, self.groups)
        self.assertEqual((len(sub_idx), len(val_idx)), (7891, 1973))
        self.assertFalse(set(self.groups[sub_idx]) & set(self.groups[val_idx]))
        preprocessor, x_sub, x_val = transform(self.frame, sub_idx, val_idx)
        self.assertEqual(x_sub.shape, (7891, 72))
        self.assertEqual(x_val.shape, (1973, 72))
        self.assertEqual(len(preprocessor.get_feature_names_out()), 72)
        self.assertTrue(np.isfinite(x_sub).all() and np.isfinite(x_val).all())

    def test_parameter_count_and_logits(self):
        model = ShoppingMLP(72, [64], 0.0)
        self.assertEqual(parameter_count(model), 4737)
        self.assertEqual(tuple(model(torch.zeros((5, 72))).shape), (5,))


if __name__ == "__main__":
    unittest.main()
