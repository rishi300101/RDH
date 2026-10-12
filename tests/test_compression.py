"""
Unit tests for src/compression.py.
"""

import os
import unittest
import numpy as np
from PIL import Image

from src.compression import (
    HUFFMAN_CODEWORDS,
    SMOOTH_BETA_TYPES,
    SMOOTH_ALPHA_TO_BETA,
    compute_block_alpha,
    map_alpha_to_beta,
    build_huffman_table,
    encode_block,
    decode_block,
    compress_error_matrix,
    decompress_error_matrix,
)
from src.predictor import predict_isgap


class TestCompression(unittest.TestCase):

    def test_alpha_calculation(self):
        # Fig. 5 example: block of size 2x2 with values [3, 2, 1, 1]
        # 3 = 0000 0011, 2 = 0000 0010, 1 = 0000 0001
        # Common prefix: 6 bits ('0000 00')
        block_fig5 = np.array([[3, 2], [1, 1]], dtype=np.uint8)
        self.assertEqual(compute_block_alpha(block_fig5), 6)

        # Identical pixels: alpha should be 8
        block_flat = np.array([[12, 12], [12, 12]], dtype=np.uint8)
        self.assertEqual(compute_block_alpha(block_flat), 8)

        # Differing MSB: alpha should be 0
        # 128 = 1000 0000, 0 = 0000 0000
        block_diff_msb = np.array([[128, 0], [0, 0]], dtype=np.uint8)
        self.assertEqual(compute_block_alpha(block_diff_msb), 0)

        # Fig. 6 block 1: [8, 4, 5, 7] -> alpha = 4
        block_f6_b1 = np.array([[8, 4], [5, 7]], dtype=np.uint8)
        self.assertEqual(compute_block_alpha(block_f6_b1), 4)

        # Fig. 6 block 2: [6, 3, 2, 17] -> alpha = 3
        block_f6_b2 = np.array([[6, 3], [2, 17]], dtype=np.uint8)
        self.assertEqual(compute_block_alpha(block_f6_b2), 3)

        # Fig. 6 block 3: [0, 1, 0, 2] -> alpha = 6
        block_f6_b3 = np.array([[0, 1], [0, 2]], dtype=np.uint8)
        self.assertEqual(compute_block_alpha(block_f6_b3), 6)

        # Fig. 6 block 4: [3, 4, 6, 4] -> alpha = 5
        block_f6_b4 = np.array([[3, 4], [6, 4]], dtype=np.uint8)
        self.assertEqual(compute_block_alpha(block_f6_b4), 5)

    def test_smooth_beta_mapping(self):
        # Explicit mapping verification from Page 5
        expected = {
            0: 0, 1: 0,
            2: 2, 3: 2,
            4: 4,
            5: 5,
            6: 6,
            7: 7, 8: 7,
        }
        for a, expected_b in expected.items():
            self.assertEqual(map_alpha_to_beta(a, category="smooth"), expected_b)

        # Verify complex category raises NotImplementedError per project instructions
        with self.assertRaises(NotImplementedError):
            map_alpha_to_beta(4, category="complex")

    def test_huffman_assignment(self):
        # Synthetic counts
        counts = {0: 10, 2: 50, 4: 100, 5: 30, 6: 5, 7: 1}
        # Order by count descending: 4 (100), 2 (50), 5 (30), 0 (10), 6 (5), 7 (1)
        table = build_huffman_table(counts)

        self.assertEqual(table[4], "00")
        self.assertEqual(table[2], "01")
        self.assertEqual(table[5], "100")
        self.assertEqual(table[0], "101")
        self.assertEqual(table[6], "110")
        self.assertEqual(table[7], "111")

        # Tie breaking: identical counts should sort by ascending beta
        tie_counts = {0: 10, 2: 10, 4: 10, 5: 10, 6: 10, 7: 10}
        tie_table = build_huffman_table(tie_counts)
        # Expected order: 0, 2, 4, 5, 6, 7
        self.assertEqual(tie_table[0], "00")
        self.assertEqual(tie_table[2], "01")
        self.assertEqual(tie_table[4], "100")
        self.assertEqual(tie_table[5], "101")
        self.assertEqual(tie_table[6], "110")
        self.assertEqual(tie_table[7], "111")

    def test_block_level_encode_decode(self):
        # Test individual block lossless round-trip across different alphas and betas
        blocks = [
            (np.array([[8, 4], [5, 7]], dtype=np.uint8), 4, "00"),
            (np.array([[6, 3], [2, 17]], dtype=np.uint8), 2, "101"),
            (np.array([[0, 1], [0, 2]], dtype=np.uint8), 6, "110"),
            (np.array([[3, 4], [6, 4]], dtype=np.uint8), 5, "01"),
            (np.array([[0, 0], [0, 0]], dtype=np.uint8), 7, "111"),
            (np.array([[200, 10], [50, 120]], dtype=np.uint8), 0, "100"),
        ]

        for block, beta, code in blocks:
            bitstream = encode_block(block, beta, code)
            reverse_dict = {code: beta}
            decoded_block, new_pos = decode_block(bitstream, 0, reverse_dict, block_size=2)
            self.assertEqual(new_pos, len(bitstream))
            self.assertTrue(np.array_equal(block, decoded_block))

    def test_paper_4x4_example_exact_match(self):
        """
        Tests the exact 4x4 matrix from Figure 6 using 2x2 blocks.
        Verifies:
        - Block 1: alpha=4, beta=4, codeword='00', bit_length=22
        - Block 2: alpha=3, beta=2, codeword='101', bit_length=29
        - Block 3: alpha=6, beta=6, codeword='110', bit_length=17
        - Block 4: alpha=5, beta=5, codeword='01', bit_length=19
        Total bit length = exactly 87 bits (matching Fig. 7).
        Lossless reconstruction = True.
        """
        P1_paper = np.array([
            [8, 4, 6, 3],
            [5, 7, 2, 17],
            [0, 1, 3, 4],
            [0, 2, 6, 4]
        ], dtype=np.uint8)

        # In Fig. 6, the author assigned:
        # beta=4 -> '00', beta=5 -> '01', beta=2 -> '101', beta=6 -> '110'
        fig6_huffman = {
            4: "00",
            5: "01",
            2: "101",
            6: "110",
            0: "100",
            7: "111",
        }

        res = compress_error_matrix(P1_paper, block_size=2, category="smooth", huffman_table=fig6_huffman)

        self.assertEqual(res["bit_length"], 87)
        self.assertEqual(res["original_bits"], 128)
        self.assertEqual(res["saved_bits"], 41)

        b_details = res["block_details"]
        # Block 1
        self.assertEqual(b_details[0]["alpha"], 4)
        self.assertEqual(b_details[0]["beta"], 4)
        self.assertEqual(b_details[0]["codeword"], "00")
        self.assertEqual(b_details[0]["bit_length"], 22)

        # Block 2
        self.assertEqual(b_details[1]["alpha"], 3)
        self.assertEqual(b_details[1]["beta"], 2)
        self.assertEqual(b_details[1]["codeword"], "101")
        self.assertEqual(b_details[1]["bit_length"], 29)

        # Block 3
        self.assertEqual(b_details[2]["alpha"], 6)
        self.assertEqual(b_details[2]["beta"], 6)
        self.assertEqual(b_details[2]["codeword"], "110")
        self.assertEqual(b_details[2]["bit_length"], 17)

        # Block 4
        self.assertEqual(b_details[3]["alpha"], 5)
        self.assertEqual(b_details[3]["beta"], 5)
        self.assertEqual(b_details[3]["codeword"], "01")
        self.assertEqual(b_details[3]["bit_length"], 19)

        # Verify exact decompression
        decomp = decompress_error_matrix(res["bitstream"], fig6_huffman, (4, 4), block_size=2)
        self.assertTrue(np.array_equal(P1_paper, decomp))

    def test_cameraman_compression_divisible_blocks(self):
        """
        Tests image-level compression and exact reconstruction on Cameraman.png
        for block sizes s=2 and s=3 (since 510x510 prediction error matrix divides evenly).
        """
        img_path = "datasets/Standard/Cameraman.png"
        self.assertTrue(os.path.exists(img_path))

        img = Image.open(img_path).convert("L")
        arr = np.array(img, dtype=np.uint8)

        # Predict using ISGAP
        pred_res = predict_isgap(arr, round_predictions=True)
        P = pred_res["best_error_matrix"]
        P1 = np.abs(P).astype(np.uint8)

        self.assertEqual(P1.shape, (510, 510))

        for s in (2, 3):
            res = compress_error_matrix(P1, block_size=s, category="smooth")
            self.assertGreater(res["compression_ratio"], 1.0)
            self.assertGreater(res["saved_bits"], 0)

            # Lossless recovery
            decomp = decompress_error_matrix(res["bitstream"], res["huffman_table"], P1.shape, block_size=s)
            self.assertTrue(np.array_equal(P1, decomp))


if __name__ == "__main__":
    unittest.main()
