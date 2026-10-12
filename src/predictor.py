"""
Predictor module for:
"Reversible data hiding and authentication scheme for encrypted image based on
prediction error compression" (Scientific Reports, 2025).

This module implements:
1. SGAP (Simplified Gradient Adjusted Predictor) - Model a1 (Eq. 1)
2. ISGAP (Improved SGAP) - 8 directional models: a1, a2, a3, a4, b1, b2, b3, b4 (Eq. 1, 2, 3)
3. Prediction error calculation: e_{i,j} = x_{i,j} - px_{i,j} (Eq. 4)
4. Model selection using total absolute error: Absumy(k) (Eq. 5)
5. MAE calculation (Eq. 11)
6. Location map (P-map) generation (Fig. 4)
"""

from typing import Dict, Tuple, Optional, Any
import numpy as np


# The eight directional model identifiers
ISGAP_MODELS = ("a1", "a2", "a3", "a4", "b1", "b2", "b3", "b4")


def round_matlab(val: np.ndarray) -> np.ndarray:
    """
    Standard round half away from zero (as implemented in MATLAB `round`).

    For pixel prediction in RDHEI schemes, discrete pixel values are required.
    MATLAB round behavior:
      round(x) = floor(x + 0.5) for x >= 0
      round(x) = ceil(x - 0.5)  for x < 0
    """
    val_arr = np.asarray(val, dtype=np.float64)
    rounded = np.where(val_arr >= 0.0, np.floor(val_arr + 0.5), np.ceil(val_arr - 0.5))
    return rounded.astype(np.int32)


def get_directional_neighbors(image: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Extracts center target pixels and the 8 surrounding neighbor matrices.

    Parameters:
        image: 2D numpy array of shape (M, N) with M, N >= 3.
               Outer borders (row 0, row M-1, col 0, col N-1) are reference pixels Pr.

    Returns:
        target: Inner pixels x of shape (M-2, N-2) corresponding to (1 <= i <= M-2, 1 <= j <= N-2).
        neighbors: Dictionary containing 2D arrays of shape (M-2, N-2):
            - 'N' : upper neighbor (i-1, j)
            - 'S' : lower neighbor (i+1, j)
            - 'W' : left neighbor (i, j-1)
            - 'E' : right neighbor (i, j+1)
            - 'NW': upper-left neighbor (i-1, j-1)
            - 'NE': upper-right neighbor (i-1, j+1)
            - 'SW': lower-left neighbor (i+1, j-1)
            - 'SE': lower-right neighbor (i+1, j+1)
    """
    if image.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image, got shape {image.shape}")
    M, N = image.shape
    if M < 3 or N < 3:
        raise ValueError(f"Image dimensions ({M}, {N}) must be at least 3x3")

    img_float = image.astype(np.float64)

    target = img_float[1 : M - 1, 1 : N - 1]
    neighbors = {
        "N": img_float[0 : M - 2, 1 : N - 1],
        "S": img_float[2:M, 1 : N - 1],
        "W": img_float[1 : M - 1, 0 : N - 2],
        "E": img_float[1 : M - 1, 2:N],
        "NW": img_float[0 : M - 2, 0 : N - 2],
        "NE": img_float[0 : M - 2, 2:N],
        "SW": img_float[2:M, 0 : N - 2],
        "SE": img_float[2:M, 2:N],
    }
    return target, neighbors


def calculate_raw_predictions(neighbors: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Calculates raw continuous predictions px for all eight ISGAP models.

    Equations from the paper:
    - a1 (SGAP, Eq. 1):  px = (xN + xW)/2 + (xNE - xNW)/4
    - b1 (Eq. 2):        px = (xN + xE)/2 + (xNW - xNE)/4
    - a2 (Eq. 3):        px = (xN + xE)/2 + (xSE - xNE)/4
    - b2 (Eq. 3):        px = (xS + xE)/2 + (xNE - xSE)/4
    - a3 (Eq. 3):        px = (xS + xE)/2 + (xSW - xSE)/4
    - b3 (Eq. 3):        px = (xS + xW)/2 + (xSE - xSW)/4
    - a4 (Eq. 3):        px = (xS + xW)/2 + (xNW - xSW)/4
    - b4 (Eq. 3):        px = (xN + xW)/2 + (xSW - xNW)/4
    """
    xN = neighbors["N"]
    xS = neighbors["S"]
    xW = neighbors["W"]
    xE = neighbors["E"]
    xNW = neighbors["NW"]
    xNE = neighbors["NE"]
    xSW = neighbors["SW"]
    xSE = neighbors["SE"]

    return {
        "a1": (xN + xW) / 2.0 + (xNE - xNW) / 4.0,
        "a2": (xN + xE) / 2.0 + (xSE - xNE) / 4.0,
        "a3": (xS + xE) / 2.0 + (xSW - xSE) / 4.0,
        "a4": (xS + xW) / 2.0 + (xNW - xSW) / 4.0,
        "b1": (xN + xE) / 2.0 + (xNW - xNE) / 4.0,
        "b2": (xS + xE) / 2.0 + (xNE - xSE) / 4.0,
        "b3": (xS + xW) / 2.0 + (xSE - xSW) / 4.0,
        "b4": (xN + xW) / 2.0 + (xSW - xNW) / 4.0,
    }


def predict_model(
    image: np.ndarray, model_name: str, round_predictions: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates predicted values using one of the eight directional models.

    Parameters:
        image: Original grayscale image (M, N).
        model_name: One of {'a1', 'a2', 'a3', 'a4', 'b1', 'b2', 'b3', 'b4'}.
        round_predictions: If True, rounds predictions to integer using MATLAB round.

    Returns:
        predicted_image: Full image of shape (M, N) where outer borders are reference
                         pixels Pr and inner area contains predicted values px.
        px_inner: Inner predicted matrix of shape (M-2, N-2).
    """
    model_name = model_name.lower()
    if model_name not in ISGAP_MODELS:
        raise ValueError(f"Unknown model '{model_name}'. Must be one of {ISGAP_MODELS}")

    target, neighbors = get_directional_neighbors(image)
    raw_preds = calculate_raw_predictions(neighbors)
    px = raw_preds[model_name]

    if round_predictions:
        px = round_matlab(px)

    predicted_image = np.array(image, dtype=px.dtype, copy=True)
    predicted_image[1 : image.shape[0] - 1, 1 : image.shape[1] - 1] = px

    return predicted_image, px


def predict_sgap(
    image: np.ndarray, round_predictions: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates predicted values using the standard SGAP model (Model a1, Eq. 1).

    Parameters:
        image: Original grayscale image (M, N).
        round_predictions: If True, rounds predictions to integer using MATLAB round.

    Returns:
        predicted_image: Full image of shape (M, N).
        px_inner: Inner predicted matrix of shape (M-2, N-2).
    """
    return predict_model(image, model_name="a1", round_predictions=round_predictions)


def calculate_prediction_error(
    image: np.ndarray, px_inner: np.ndarray
) -> np.ndarray:
    """
    Calculates prediction error matrix P of size (M-2) x (N-2) according to Eq. (4):
      e_{i,j} = x_{i,j} - px_{i,j}

    Parameters:
        image: Original image of shape (M, N).
        px_inner: Inner predicted matrix of shape (M-2, N-2).

    Returns:
        error_matrix: Integer prediction error matrix of shape (M-2, N-2).
    """
    x_inner = image[1 : image.shape[0] - 1, 1 : image.shape[1] - 1]
    return (x_inner - px_inner).astype(np.int32)


def calculate_absumy(error_matrix: np.ndarray) -> float:
    """
    Calculates the sum of absolute values of all elements in an error matrix
    according to Eq. (5):
      Absumy(k) = sum_{i=2}^{M-1} sum_{j=2}^{N-1} |e_{i,j}|
    """
    return float(np.sum(np.abs(error_matrix)))


def calculate_mae(error_matrix: np.ndarray) -> float:
    """
    Calculates the Mean Absolute Error (MAE) according to Eq. (11):
      MAE = (1 / n) * sum_{y=1}^{n} |y_true - y_pred|
    where n = (M-2) * (N-2) is the number of predicted samples.
    """
    return float(np.mean(np.abs(error_matrix)))


def generate_location_map(error_matrix: np.ndarray) -> np.ndarray:
    """
    Generates location map (P-map) of the same size as error matrix P (Fig. 4).
    As shown in Fig. 4 of the paper:
      P-map is 1 where error is negative (e_{i,j} < 0),
      and 0 where error is non-negative (e_{i,j} >= 0).

    Parameters:
        error_matrix: Prediction error matrix of shape (M-2, N-2).

    Returns:
        p_map: Binary location map of shape (M-2, N-2), dtype uint8.
    """
    return np.where(error_matrix < 0, 1, 0).astype(np.uint8)


def predict_isgap(
    image: np.ndarray, round_predictions: bool = True
) -> Dict[str, Any]:
    """
    Executes the full ISGAP predictor algorithm across all eight directional models.

    Evaluates models: a1, a2, a3, a4, b1, b2, b3, b4.
    Selects the optimal model that minimizes Absumy(k) (Eq. 5).

    Parameters:
        image: Original grayscale image (M, N).
        round_predictions: If True, uses discrete integer predictions (MATLAB round).

    Returns:
        dict containing:
            - 'best_model': Name of the best model (e.g., 'a1', 'a4', 'b1', ...)
            - 'best_pred_image': Full predicted image (M, N) for the best model
            - 'best_error_matrix': Prediction error matrix P (M-2, N-2)
            - 'best_absumy': Total absolute error of the best model
            - 'best_mae': Mean Absolute Error of the best model
            - 'best_p_map': P-map location map (M-2, N-2)
            - 'sgap_mae': MAE of the baseline SGAP (model a1)
            - 'sgap_absumy': Absumy of baseline SGAP
            - 'models_eval': Dict mapping each model name to its Absumy and MAE
    """
    target, neighbors = get_directional_neighbors(image)
    raw_preds = calculate_raw_predictions(neighbors)

    models_eval = {}
    best_model = None
    min_absumy = float("inf")
    best_px = None
    best_error = None

    for m in ISGAP_MODELS:
        px = raw_preds[m]
        if round_predictions:
            px = round_matlab(px)

        error_matrix = (target - px).astype(np.int32)
        absumy = calculate_absumy(error_matrix)
        mae = calculate_mae(error_matrix)

        models_eval[m] = {
            "absumy": absumy,
            "mae": mae,
        }

        if absumy < min_absumy:
            min_absumy = absumy
            best_model = m
            best_px = px
            best_error = error_matrix

    # Full predicted image for the optimal model
    best_pred_image = np.array(image, dtype=best_px.dtype, copy=True)
    best_pred_image[1 : image.shape[0] - 1, 1 : image.shape[1] - 1] = best_px

    best_p_map = generate_location_map(best_error)
    sgap_mae = models_eval["a1"]["mae"]
    sgap_absumy = models_eval["a1"]["absumy"]

    return {
        "best_model": best_model,
        "best_pred_image": best_pred_image,
        "best_error_matrix": best_error,
        "best_absumy": min_absumy,
        "best_mae": models_eval[best_model]["mae"],
        "best_p_map": best_p_map,
        "sgap_mae": sgap_mae,
        "sgap_absumy": sgap_absumy,
        "models_eval": models_eval,
    }


if __name__ == "__main__":
    import os
    import sys
    from PIL import Image

    # Default test image path or user-supplied command line argument
    test_path = sys.argv[1] if len(sys.argv) > 1 else "datasets/Standard/Cameraman.png"

    if not os.path.exists(test_path):
        print(f"Error: Image file not found at '{test_path}'")
        sys.exit(1)

    print(f"Loading test image: {test_path}")
    img = Image.open(test_path).convert("L")
    arr = np.array(img, dtype=np.uint8)
    M, N = arr.shape
    print(f"Image dimensions: {M} x {N}")

    results = predict_isgap(arr, round_predictions=True)

    print("-" * 55)
    print(f"SGAP (Model a1)  MAE: {results['sgap_mae']:.3f}")
    print(f"Optimal Model       : {results['best_model']}")
    print(f"ISGAP Best Model MAE: {results['best_mae']:.3f}")
    reduction = ((results["sgap_mae"] - results["best_mae"]) / results["sgap_mae"]) * 100
    print(f"MAE Reduction       : {reduction:.2f}%")
    print("-" * 55)
    print("Evaluation of all 8 models:")
    for m in ISGAP_MODELS:
        mae = results["models_eval"][m]["mae"]
        absumy = results["models_eval"][m]["absumy"]
        marker = " <== (Optimal)" if m == results["best_model"] else ""
        print(f"  Model {m:2s}: MAE = {mae:6.3f}, Absumy = {absumy:10.0f}{marker}")
    print("-" * 55)
