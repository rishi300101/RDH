"""
Tests for src/predictor.py
"""

import unittest
import numpy as np
from src.predictor import (
    ISGAP_MODELS,
    round_matlab,
    get_directional_neighbors,
    calculate_raw_predictions,
    predict_sgap,
    predict_model,
    predict_isgap,
    calculate_prediction_error,
    calculate_absumy,
    calculate_mae,
    generate_location_map,
)


class TestPredictor(unittest.TestCase):
    def test_round_matlab(self):
        self.assertEqual(round_matlab(np.array([0.5]))[0], 1)
        self.assertEqual(round_matlab(np.array([0.25]))[0], 0)
        self.assertEqual(round_matlab(np.array([0.75]))[0], 1)
        self.assertEqual(round_matlab(np.array([-0.5]))[0], -1)
        self.assertEqual(round_matlab(np.array([-0.25]))[0], 0)

    def test_synthetic_prediction(self):
        # 4x4 flat image
        img = np.array([
            [10, 10, 10, 10],
            [10, 10, 10, 10],
            [10, 10, 10, 10],
            [10, 10, 10, 10],
        ], dtype=np.uint8)

        res = predict_isgap(img)
        self.assertEqual(res["best_absumy"], 0.0)
        self.assertEqual(res["best_mae"], 0.0)
        self.assertTrue(np.all(res["best_error_matrix"] == 0))
        self.assertTrue(np.all(res["best_p_map"] == 0))

    def test_error_and_pmap(self):
        # Test Fig. 4 example from paper
        err = np.array([
            [-8, 4, 6, 3],
            [5, -7, 2, -17],
            [0, -1, 3, -4],
            [0, 2, 6, 4]
        ], dtype=np.int32)

        pmap = generate_location_map(err)
        expected_pmap = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 1],
            [0, 1, 0, 1],
            [0, 0, 0, 0]
        ], dtype=np.uint8)

        self.assertTrue(np.array_equal(pmap, expected_pmap))


if __name__ == "__main__":
    unittest.main()
