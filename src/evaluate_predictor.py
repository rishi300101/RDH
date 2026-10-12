"""
Evaluation script for RDHEI Predictor Module (Stage 1).

Evaluates all 12 images in datasets/Standard:
- Calculates MAE for SGAP and all eight ISGAP models (a1..b4).
- Selects the optimal ISGAP model minimizing Absumy.
- Saves results to results/predictor_mae.csv.
- Saves visual comparisons for Baboon, Boat, and Cameraman in results/predictions/.
- Saves error maps for those images in results/error_maps/.
"""

import os
import sys
import csv
import hashlib

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from src.predictor import (
    ISGAP_MODELS,
    predict_isgap,
)


STANDARD_DIR = "datasets/Standard"
RESULTS_DIR = "results"
PREDICTIONS_DIR = os.path.join(RESULTS_DIR, "predictions")
ERROR_MAPS_DIR = os.path.join(RESULTS_DIR, "error_maps")
CSV_PATH = os.path.join(RESULTS_DIR, "predictor_mae.csv")

TARGET_VISUAL_IMAGES = ["Baboon.png", "Boat.png", "Cameraman.png"]


def evaluate_all_standard_images():
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    os.makedirs(ERROR_MAPS_DIR, exist_ok=True)

    # Gather standard images
    image_files = sorted([f for f in os.listdir(STANDARD_DIR) if f.lower().endswith((".png", ".jpg", ".bmp", ".pgm"))])
    print(f"Found {len(image_files)} images in {STANDARD_DIR}: {image_files}")

    csv_rows = []
    summary_stats = []
    seen_pixel_hashes = {}  # sha256 -> first_image_name

    for filename in image_files:
        img_path = os.path.join(STANDARD_DIR, filename)
        img = Image.open(img_path).convert("L")
        img_arr = np.array(img, dtype=np.uint8)

        # Check for pixel equality duplicates
        pixel_hash = hashlib.sha256(img_arr.tobytes()).hexdigest()
        img_base = os.path.splitext(filename)[0]
        if pixel_hash in seen_pixel_hashes:
            is_duplicate = True
            duplicate_of = seen_pixel_hashes[pixel_hash]
        else:
            is_duplicate = False
            duplicate_of = ""
            seen_pixel_hashes[pixel_hash] = img_base

        # Evaluate using ISGAP
        res = predict_isgap(img_arr, round_predictions=True)

        sgap_mae = res["sgap_mae"]
        best_model = res["best_model"]
        best_mae = res["best_mae"]
        reduction_pct = ((sgap_mae - best_mae) / sgap_mae * 100.0) if sgap_mae > 0 else 0.0

        row = {
            "image": img_base,
            "sgap_mae": round(sgap_mae, 4),
            "a1_mae": round(res["models_eval"]["a1"]["mae"], 4),
            "a2_mae": round(res["models_eval"]["a2"]["mae"], 4),
            "a3_mae": round(res["models_eval"]["a3"]["mae"], 4),
            "a4_mae": round(res["models_eval"]["a4"]["mae"], 4),
            "b1_mae": round(res["models_eval"]["b1"]["mae"], 4),
            "b2_mae": round(res["models_eval"]["b2"]["mae"], 4),
            "b3_mae": round(res["models_eval"]["b3"]["mae"], 4),
            "b4_mae": round(res["models_eval"]["b4"]["mae"], 4),
            "best_model": best_model,
            "best_mae": round(best_mae, 4),
            "reduction_percent": round(reduction_pct, 2),
            "is_duplicate": is_duplicate,
            "duplicate_of": duplicate_of,
        }
        csv_rows.append(row)
        summary_stats.append(row)

        dup_tag = f" [Duplicate of {duplicate_of}]" if is_duplicate else ""
        print(
            f"Processed {img_base:10s} | SGAP: {sgap_mae:6.3f} | Best: {best_model:2s} ({best_mae:6.3f}) | Red: {reduction_pct:5.2f}%{dup_tag}"
        )

        # Generate visual outputs for Baboon, Boat, Cameraman
        if filename in TARGET_VISUAL_IMAGES:
            pred_img = np.clip(res["best_pred_image"], 0, 255).astype(np.uint8)
            err_matrix = np.abs(res["best_error_matrix"])

            # 1. Save predicted image standalone
            pred_path = os.path.join(PREDICTIONS_DIR, f"{img_base}_predicted_{best_model}.png")
            Image.fromarray(pred_img).save(pred_path)

            # 2. Save side-by-side visual comparison
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(img_arr, cmap="gray", vmin=0, vmax=255)
            axes[0].set_title(f"Original ({img_base})", fontsize=12)
            axes[0].axis("off")

            axes[1].imshow(pred_img, cmap="gray", vmin=0, vmax=255)
            axes[1].set_title(f"Predicted (ISGAP Model {best_model})\nMAE: {best_mae:.3f}", fontsize=12)
            axes[1].axis("off")

            plt.tight_layout()
            comp_path = os.path.join(PREDICTIONS_DIR, f"{img_base}_comparison.png")
            plt.savefig(comp_path, dpi=200, bbox_inches="tight")
            plt.close(fig)

            # 3. Save absolute prediction-error map
            fig, ax = plt.subplots(figsize=(6, 6))
            cax = ax.imshow(err_matrix, cmap="hot", interpolation="nearest")
            ax.set_title(f"Absolute Prediction Error Map\n{img_base} (Model {best_model}, MAE={best_mae:.3f})", fontsize=11)
            ax.axis("off")
            fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04, label="|e(i, j)|")
            plt.tight_layout()
            err_map_path = os.path.join(ERROR_MAPS_DIR, f"{img_base}_abs_error_map.png")
            plt.savefig(err_map_path, dpi=200, bbox_inches="tight")
            plt.close(fig)

            # Also save raw uint8 normalized error map
            max_err = max(float(np.max(err_matrix)), 1.0)
            err_norm = (err_matrix / max_err * 255.0).astype(np.uint8)
            raw_err_path = os.path.join(ERROR_MAPS_DIR, f"{img_base}_error_grayscale.png")
            Image.fromarray(err_norm).save(raw_err_path)

    # Write CSV
    fieldnames = [
        "image",
        "sgap_mae",
        "a1_mae",
        "a2_mae",
        "a3_mae",
        "a4_mae",
        "b1_mae",
        "b2_mae",
        "b3_mae",
        "b4_mae",
        "best_model",
        "best_mae",
        "reduction_percent",
        "is_duplicate",
        "duplicate_of",
    ]
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\nSaved CSV to: {CSV_PATH}")

    # Summary statistics: All 12 files
    avg_sgap_all = np.mean([r["sgap_mae"] for r in summary_stats])
    avg_isgap_all = np.mean([r["best_mae"] for r in summary_stats])
    avg_red_all = np.mean([r["reduction_percent"] for r in summary_stats])

    # Summary statistics: 11 Unique images (excluding only the duplicate copy)
    unique_stats = [r for r in summary_stats if not r["is_duplicate"]]
    avg_sgap_unique = np.mean([r["sgap_mae"] for r in unique_stats])
    avg_isgap_unique = np.mean([r["best_mae"] for r in unique_stats])
    avg_red_unique = np.mean([r["reduction_percent"] for r in unique_stats])

    print("\n" + "=" * 65)
    print("EVALUATION SUMMARY STATISTICS")
    print("=" * 65)
    print("1. ALL 12 FILES (including duplicate Sailboat.png):")
    print(f"   - Average SGAP MAE       : {avg_sgap_all:.4f}")
    print(f"   - Average ISGAP Best MAE : {avg_isgap_all:.4f}")
    print(f"   - Average MAE Reduction  : {avg_red_all:.2f}%")
    print("-" * 65)
    print("2. 11 UNIQUE IMAGES (excluding duplicate copy Sailboat.png):")
    print(f"   - Average SGAP MAE       : {avg_sgap_unique:.4f}")
    print(f"   - Average ISGAP Best MAE : {avg_isgap_unique:.4f}")
    print(f"   - Average MAE Reduction  : {avg_red_unique:.2f}%")
    print("-" * 65)

    model_counts_all = {}
    for r in summary_stats:
        m = r["best_model"]
        model_counts_all[m] = model_counts_all.get(m, 0) + 1
    model_counts_unique = {}
    for r in unique_stats:
        m = r["best_model"]
        model_counts_unique[m] = model_counts_unique.get(m, 0) + 1

    print(f"Optimal Model Distribution (All 12)   : {dict(sorted(model_counts_all.items()))}")
    print(f"Optimal Model Distribution (11 Unique): {dict(sorted(model_counts_unique.items()))}")
    print("=" * 65)



if __name__ == "__main__":
    evaluate_all_standard_images()
