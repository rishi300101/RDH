"""
Prediction-Error Compression module for:
"Reversible data hiding and authentication scheme for encrypted image based on
prediction error compression" (Scientific Reports, 2025).

This module implements the base paper's prediction-error compression algorithm:
1. Block alpha calculation (common MSB prefix length of 8-bit unsigned error values).
2. Block type merging for smooth images (beta in {0, 2, 4, 5, 6, 7}).
3. Adaptive Huffman coding of merged block types using fixed codewords:
   {'00', '01', '100', '101', '110', '111'}.
4. Block-by-block bitstream encoding and lossless decoding.
5. Error matrix level compression, decompression, and metrics calculation.

Note on Scope and Limitations:
- Only the paper's explicitly specified smooth-image mapping is implemented.
  The complex-image mapping is not fully specified in the paper (mapping for types
  {1, 6, 8} is omitted) and is rejected to avoid guessing.
- Dimensions must be evenly divisible by the block size s. The paper provides
  no algorithm for handling boundary pixels when (M-2) is not divisible by s.
- Payload bit counts and compression ratios measure the compressed error blocks
  and exclude the sign map (P-map) and header metadata, as identified in Section 3.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np


# The six fixed canonical Huffman codewords specified in the paper (Page 5)
HUFFMAN_CODEWORDS = ("00", "01", "100", "101", "110", "111")

# The six valid merged types for smooth images (Page 5)
SMOOTH_BETA_TYPES = (0, 2, 4, 5, 6, 7)

# Explicit smooth-image mapping from alpha in {0..8} to beta in {0, 2, 4, 5, 6, 7}
SMOOTH_ALPHA_TO_BETA: Dict[int, int] = {
    0: 0,
    1: 0,
    2: 2,
    3: 2,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 7,
}


def compute_block_alpha(block: np.ndarray) -> int:
    """
    Computes block type alpha in {0, 1, ..., 8} for an s x s error block.

    According to the paper (Page 5, Fig. 5):
    - Values are treated as 8-bit unsigned binary sequences (MSB bit 7 to LSB bit 0).
    - The reference error e1 (first pixel) and remaining s^2 - 1 error values
      are compared successively from MSB to LSB.
    - alpha is the length of the identical bit prefix shared by ALL error values
      in the block (called the flag bit).

    Parameters:
        block: 2D or 1D array of non-negative integer error values in [0, 255].

    Returns:
        alpha: Integer in range [0, 8].
    """
    flat = np.asarray(block, dtype=np.int32).flatten()
    if len(flat) == 0:
        raise ValueError("Cannot compute alpha for an empty block.")
    if np.any((flat < 0) | (flat > 255)):
        raise ValueError("Error values must be in the range [0, 255].")

    ref = flat[0]
    alpha = 0
    # Compare bit-by-bit from MSB (bit 7) down to LSB (bit 0)
    for bit_pos in range(7, -1, -1):
        ref_bit = (ref >> bit_pos) & 1
        # Check if all pixels have the exact same bit at bit_pos
        all_match = True
        for val in flat[1:]:
            if ((val >> bit_pos) & 1) != ref_bit:
                all_match = False
                break
        if all_match:
            alpha += 1
        else:
            break

    return alpha


def map_alpha_to_beta(alpha: int, category: str = "smooth") -> int:
    """
    Maps original block type alpha in {0..8} to merged block type beta.

    For smooth images (explicitly specified in paper, Page 5):
      types 0, 1 -> 0
      types 2, 3 -> 2
      type 4     -> 4
      type 5     -> 5
      type 6     -> 6
      types 7, 8 -> 7
    Resulting set: beta in {0, 2, 4, 5, 6, 7}.

    For complex images:
      The paper only states beta_c = {0, 2, 3, 4, 5, 7} but does not specify
      the mapping for {1, 6, 8}. In accordance with project instructions,
      complex mapping is not implemented using an inferred rule.

    Parameters:
        alpha: Integer in range [0, 8].
        category: "smooth" (only supported category).

    Returns:
        beta: Integer merged block type.
    """
    if alpha not in SMOOTH_ALPHA_TO_BETA:
        raise ValueError(f"Invalid alpha value {alpha}. Must be in range 0..8.")

    if category.lower() == "smooth":
        return SMOOTH_ALPHA_TO_BETA[alpha]
    elif category.lower() == "complex":
        raise NotImplementedError(
            "Complex-image mapping is ambiguous in the base paper. The target set "
            "beta_c = {0, 2, 3, 4, 5, 7} is stated on Page 5, but the specific "
            "mapping for original types {1, 6, 8} is omitted. Per project guidelines, "
            "an inferred mapping is not implemented."
        )
    else:
        raise ValueError(f"Unknown category '{category}'. Must be 'smooth'.")


def build_huffman_table(
    beta_counts: Dict[int, int],
    valid_betas: Tuple[int, ...] = SMOOTH_BETA_TYPES,
) -> Dict[int, str]:
    """
    Builds adaptive Huffman code table assigning the six fixed codewords:
      {'00', '01', '100', '101', '110', '111'}
    to the six merged block types based on their occurrence frequency (Page 5).

    The paper states:
      "the shorter code marks the type value with the larger number of blocks
       and the longer code marks the type value with the fewer number of blocks."

    Deterministic Tie-Breaking Rule:
      The paper does not specify a tie-breaking rule when types have identical
      counts. Here, ties are broken deterministically by ascending numerical
      value of beta (e.g., type 0 before type 2).

    Parameters:
        beta_counts: Dictionary mapping merged type beta to its occurrence count.
        valid_betas: The set of 6 merged block types (default: SMOOTH_BETA_TYPES).

    Returns:
        huffman_table: Dict mapping each beta -> codeword string.
    """
    # Ensure all valid types are present in counts
    full_counts = {b: beta_counts.get(b, 0) for b in valid_betas}

    # Sort primarily by count descending, secondarily by beta ascending
    sorted_betas = sorted(full_counts.keys(), key=lambda b: (-full_counts[b], b))

    # Assign codewords in fixed order
    table = {}
    for beta, code in zip(sorted_betas, HUFFMAN_CODEWORDS):
        table[beta] = code

    return table


def encode_block(
    block: np.ndarray,
    beta: int,
    codeword: str,
) -> str:
    """
    Encodes a single s x s error block into its binary bitstream string.

    According to the paper (Page 5, Step 2 & Fig. 7):
    1. Block type code: Huffman codeword for beta (2 or 3 bits).
    2. Reference error value e1: The full 8-bit unsigned value of the first pixel.
    3. Remaining s^2 - 1 error values: The (8 - beta) least significant bits (LSB)
       of each remaining pixel.

    Parameters:
        block: 2D or 1D array of shape with s^2 elements in [0, 255].
        beta: Merged block type in {0, 2, 4, 5, 6, 7}.
        codeword: Huffman codeword string (e.g., '00', '101').

    Returns:
        bitstream: String of '0' and '1' representing the compressed block.
    """
    flat = np.asarray(block, dtype=np.int32).flatten()
    e1 = int(flat[0])
    num_lsb_bits = 8 - beta

    bits = [codeword]
    # Store full 8-bit value of reference pixel
    bits.append(format(e1, "08b"))

    # Store (8 - beta) LSB bits for each remaining pixel
    if num_lsb_bits > 0:
        mask = (1 << num_lsb_bits) - 1
        for val in flat[1:]:
            lsb_val = int(val) & mask
            bits.append(format(lsb_val, f"0{num_lsb_bits}b"))

    return "".join(bits)


def decode_block(
    bitstream: str,
    start_pos: int,
    reverse_huffman: Dict[str, int],
    block_size: int,
) -> Tuple[np.ndarray, int]:
    """
    Decodes a single s x s error block from the compressed bitstream.

    According to the paper (Page 5, Section 'Block type merging and encoding'):
    1. Read Huffman codeword -> merged type beta.
    2. Read 8 bits -> reference error e1.
    3. Reconstruct remaining s^2 - 1 pixels:
       All pixels share the same beta MSB prefix as e1. Combine the shared
       prefix with the read (8 - beta) LSB bits.

    Parameters:
        bitstream: Full compressed binary string.
        start_pos: Current reading index in bitstream.
        reverse_huffman: Dict mapping codeword string -> beta integer.
        block_size: Block dimension s (block has s x s pixels).

    Returns:
        decoded_block: 2D numpy array of shape (s, s), dtype uint8.
        new_pos: Updated reading index in bitstream.
    """
    # 1. Match Huffman codeword
    matched_codeword = None
    for length in (2, 3):
        candidate = bitstream[start_pos : start_pos + length]
        if candidate in reverse_huffman:
            matched_codeword = candidate
            break

    if matched_codeword is None:
        raise ValueError(f"Could not parse valid Huffman codeword at position {start_pos}")

    beta = reverse_huffman[matched_codeword]
    pos = start_pos + len(matched_codeword)

    # 2. Read 8-bit reference error e1
    e1 = int(bitstream[pos : pos + 8], 2)
    pos += 8

    # 3. Reconstruct s^2 pixels
    num_pixels = block_size * block_size
    num_lsb_bits = 8 - beta
    reconstructed = [e1]

    # The shared prefix is the top beta bits of e1
    prefix_mask = ((1 << beta) - 1) << (8 - beta) if beta > 0 else 0
    prefix = e1 & prefix_mask

    for _ in range(num_pixels - 1):
        if num_lsb_bits > 0:
            lsb_val = int(bitstream[pos : pos + num_lsb_bits], 2)
            pos += num_lsb_bits
            pixel_val = prefix | lsb_val
        else:
            pixel_val = prefix
        reconstructed.append(pixel_val)

    decoded_arr = np.array(reconstructed, dtype=np.uint8).reshape((block_size, block_size))
    return decoded_arr, pos


def compress_error_matrix(
    P1: np.ndarray,
    block_size: int = 2,
    category: str = "smooth",
    huffman_table: Optional[Dict[int, str]] = None,
) -> Dict[str, Any]:
    """
    Compresses an absolute prediction error matrix P1 block-by-block.

    Parameters:
        P1: 2D numpy array of absolute prediction errors in [0, 255].
        block_size: Side length s of each square block.
        category: "smooth" (only supported category).
        huffman_table: Optional pre-defined Huffman table. If None, built adaptively.

    Returns:
        dict containing:
            - 'bitstream': Compressed binary string of all blocks.
            - 'bit_length': Length of compressed bitstream in bits.
            - 'original_bits': Total uncompressed bits of the matrix (H * W * 8).
            - 'compression_ratio': CR = original_bits / bit_length (Eq. 12).
            - 'saved_bits': original_bits - bit_length.
            - 'huffman_table': Dictionary of Huffman codes used.
            - 'block_details': List of dicts with details for each block.
            - 'shape': Shape of input matrix (H, W).
            - 'block_size': Block size s.
    """
    if P1.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {P1.shape}")

    H, W = P1.shape
    s = block_size

    # Strict boundary check: paper does not specify boundary pixel handling
    if H % s != 0 or W % s != 0:
        raise ValueError(
            f"Matrix dimensions ({H}, {W}) are not divisible by block size s={s}. "
            f"Leftover rows ({H % s}) or columns ({W % s}) cannot be handled without "
            f"guessing boundary rules omitted in the base paper."
        )

    num_blocks_h = H // s
    num_blocks_w = W // s

    # First pass: compute alpha and beta for all blocks
    blocks_data = []
    beta_counts = {b: 0 for b in SMOOTH_BETA_TYPES}

    for bh in range(num_blocks_h):
        for bw in range(num_blocks_w):
            r_start = bh * s
            c_start = bw * s
            sub_block = P1[r_start : r_start + s, c_start : c_start + s]
            alpha = compute_block_alpha(sub_block)
            beta = map_alpha_to_beta(alpha, category=category)
            beta_counts[beta] += 1
            blocks_data.append({
                "sub_block": sub_block,
                "alpha": alpha,
                "beta": beta,
            })

    # Second pass: build Huffman table if not provided
    if huffman_table is None:
        huffman_table = build_huffman_table(beta_counts)

    # Third pass: encode all blocks
    bitstream_parts = []
    block_details = []

    for idx, b_info in enumerate(blocks_data):
        sub_block = b_info["sub_block"]
        alpha = b_info["alpha"]
        beta = b_info["beta"]
        codeword = huffman_table[beta]

        block_bits = encode_block(sub_block, beta, codeword)
        bitstream_parts.append(block_bits)

        block_details.append({
            "block_index": idx,
            "alpha": alpha,
            "beta": beta,
            "codeword": codeword,
            "bit_length": len(block_bits),
            "ref_pixel": int(sub_block[0, 0]),
        })

    full_bitstream = "".join(bitstream_parts)
    original_bits = H * W * 8
    compressed_bits = len(full_bitstream)
    cr = original_bits / compressed_bits if compressed_bits > 0 else 0.0
    saved_bits = original_bits - compressed_bits

    return {
        "bitstream": full_bitstream,
        "bit_length": compressed_bits,
        "original_bits": original_bits,
        "compression_ratio": cr,
        "saved_bits": saved_bits,
        "huffman_table": huffman_table,
        "beta_counts": beta_counts,
        "block_details": block_details,
        "shape": (H, W),
        "block_size": s,
    }


def decompress_error_matrix(
    bitstream: str,
    huffman_table: Dict[int, str],
    shape: Tuple[int, int],
    block_size: int,
) -> np.ndarray:
    """
    Losslessly decompresses the bitstream back into the original error matrix P1.

    Parameters:
        bitstream: Compressed binary string.
        huffman_table: Dict mapping beta -> codeword string.
        shape: Dimensions (H, W) of the target matrix.
        block_size: Block dimension s.

    Returns:
        P1_reconstructed: 2D numpy array of shape (H, W), dtype uint8.
    """
    H, W = shape
    s = block_size
    num_blocks_h = H // s
    num_blocks_w = W // s

    reverse_huffman = {code: beta for beta, code in huffman_table.items()}

    reconstructed_matrix = np.zeros((H, W), dtype=np.uint8)
    pos = 0

    for bh in range(num_blocks_h):
        for bw in range(num_blocks_w):
            decoded_block, pos = decode_block(bitstream, pos, reverse_huffman, s)
            r_start = bh * s
            c_start = bw * s
            reconstructed_matrix[r_start : r_start + s, c_start : c_start + s] = decoded_block

    if pos != len(bitstream):
        raise ValueError(f"Decompression did not consume all bits: {pos}/{len(bitstream)}")

    return reconstructed_matrix
