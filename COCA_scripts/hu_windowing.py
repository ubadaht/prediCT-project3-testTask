"""
hu_windowing.py
───────────────
HU Windowing for Cardiac Calcium Scoring CT.

Window Center (WC) = 400 HU
Window Width  (WW) = 1000 HU
→ Clips to [-100, 900] HU then normalizes to [0.0, 1.0]

Usage:
    from hu_windowing import apply_window, verify_windowing

Reference: Ubadah Tanveer 2026
"""

import numpy as np
import SimpleITK as sitk
from pathlib import Path


# ── Constants ──────────────────────────────────────────────────────────────────
WINDOW_CENTER = 400   # HU  — centered on calcium/bone range
WINDOW_WIDTH  = 1000  # HU  — wide enough to keep soft tissue context

HU_MIN = WINDOW_CENTER - WINDOW_WIDTH // 2   # -100 HU
HU_MAX = WINDOW_CENTER + WINDOW_WIDTH // 2   #  900 HU


def apply_window(
    image: np.ndarray,
    hu_min: float = HU_MIN,
    hu_max: float = HU_MAX
) -> np.ndarray:
    """
    Applies HU windowing to a 3D CT numpy array.

    Steps:
        1. Clip values to [hu_min, hu_max]
        2. Normalize to [0.0, 1.0]

    Args:
        image  : numpy array of shape (Z, Y, X) with raw HU values
        hu_min : lower HU bound (default -100)
        hu_max : upper HU bound (default  900)

    Returns:
        numpy array of same shape, dtype float32, values in [0.0, 1.0]
    """
    # Step 1 — Clip
    windowed = np.clip(image, hu_min, hu_max)

    # Step 2 — Normalize to [0, 1]
    windowed = (windowed - hu_min) / (hu_max - hu_min)

    return windowed.astype(np.float32)


def window_from_sitk(image: sitk.Image) -> np.ndarray:
    """
    Convenience wrapper: takes a SimpleITK image, returns windowed numpy array.

    Args:
        image : SimpleITK image loaded from .nii.gz

    Returns:
        windowed numpy array, float32, [0.0, 1.0]
    """
    arr = sitk.GetArrayFromImage(image)   # shape: (Z, Y, X)
    return apply_window(arr)


# ── Verification ───────────────────────────────────────────────────────────────

def verify_windowing(resampled_dir: str, n_samples: int = 5):
    """
    To Spot-checks windowing on n_samples scans from the resampled directory.
    Prints before/after statistics to confirm windowing is working correctly.

    Args:
        resampled_dir : path to COCA_output/data_resampled/
        n_samples     : how many scans to check (default 5)
    """
    resampled_path = Path(resampled_dir)
    scan_folders   = [f for f in resampled_path.iterdir() if f.is_dir()]

    if not scan_folders:
        print(f"[ERROR] No scan folders found in {resampled_dir}")
        return

    # Pick n_samples evenly spaced across the dataset
    indices  = np.linspace(0, len(scan_folders) - 1, n_samples, dtype=int)
    samples  = [scan_folders[i] for i in indices]

    print(f"Verifying HU windowing on {n_samples} scans...\n")
    print(f"{'Scan ID':<15} {'Raw Min':>8} {'Raw Max':>8} {'Raw Mean':>9} │ "
          f"{'Win Min':>8} {'Win Max':>8} {'Win Mean':>9}")
    print("─" * 75)

    for folder in samples:
        scan_id  = folder.name
        img_path = folder / f"{scan_id}_img.nii.gz"

        if not img_path.exists():
            print(f"{scan_id:<15} [FILE NOT FOUND]")
            continue

        # Load raw
        image    = sitk.ReadImage(str(img_path))
        raw_arr  = sitk.GetArrayFromImage(image).astype(np.float32)

        # Apply window
        win_arr  = apply_window(raw_arr)

        print(f"{scan_id:<15} "
              f"{raw_arr.min():>8.1f} {raw_arr.max():>8.1f} {raw_arr.mean():>9.2f} │ "
              f"{win_arr.min():>8.4f} {win_arr.max():>8.4f} {win_arr.mean():>9.4f}")

    print("\nExpected after windowing:")
    print(f"  Min  → 0.0000  (was anything ≤ {HU_MIN} HU)")
    print(f"  Max  → 1.0000  (was anything ≥ {HU_MAX} HU)")
    print(f"  Mean → ~0.10 to 0.35  (mostly soft tissue + some calcium)")
    print(f"\nWindow applied: [{HU_MIN}, {HU_MAX}] HU  "
          f"(Center={WINDOW_CENTER}, Width={WINDOW_WIDTH})")



if __name__ == "__main__":
    RESAMPLED_DIR = r"...\COCA_output\data_resampled"
    verify_windowing(RESAMPLED_DIR, n_samples=5)
