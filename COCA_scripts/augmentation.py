"""
Data Augmentation Pipeline for Cardiac Calcium Scoring CT.
Usage:
    from augmentation import get_train_transforms, get_val_transforms
"""

from monai.transforms import (
    Compose,
    # Loading & formatting
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    # Spatial
    RandFlipd,
    RandRotated,
    RandZoomd,
    # Intensity
    RandGaussianNoised,
    RandScaleIntensityd,
    # Utility
    ScaleIntensityRanged,
    
)
import numpy as np


# HU Window Constants (must match hu_windowing.py)
HU_MIN = -100.0
HU_MAX =  900.0


#  Transform Factories

def get_train_transforms():
    """
    Full augmentation pipeline for TRAINING set.

      1. Load & format first
      2. HU windowing
      3. Geometric augmentation
      4. Intensity augmentation (random, applied to image only

    Returns:
        MONAI Compose object
    """
    return Compose([

        #  Step 1: Load NIfTI files
        LoadImaged(keys=["image", "mask"], image_only=True),
        EnsureChannelFirstd(keys=["image", "mask"]),  # (Z,Y,X) → (1,Z,Y,X)
        EnsureTyped(keys=["image", "mask"], dtype=np.float32),

        #  Step 2: HU Windowing
        # Clips raw HU to [-100, 900] then normalizes to [0, 1]
        ScaleIntensityRanged(
            keys=["image"],
            a_min=HU_MIN,
            a_max=HU_MAX,
            b_min=0.0,
            b_max=1.0,
            clip=True          # clips values outside [a_min, a_max]
        ),

        #  Step 3: Geometric Augmentation 
        # Applied IDENTICALLY to both image and mask

        # Left/Right flip only (axis=0 in channel-first = spatial axis Z)
        # For cardiac CT: axis=2 is left/right (X axis)
        RandFlipd(
            keys=["image", "mask"],
            spatial_axis=2,     # X axis = left/right
            prob=0.50
        ),

        # Small rotations to simulate patient positioning variability
        # range_x/y/z in radians: 15° = 0.2618 rad
        RandRotated(
            keys=["image", "mask"],
            range_x=0.2618,     # ±15° around X axis
            range_y=0.2618,     # ±15° around Y axis
            range_z=0.2618,     # ±15° around Z axis
            prob=0.50,
            keep_size=True,     # output same size as input
            mode=["bilinear", "nearest"],  # bilinear for image, nearest for mask
            padding_mode="zeros"
        ),

        # Zoom to simulate body habitus variation
        RandZoomd(
            keys=["image", "mask"],
            min_zoom=0.85,
            max_zoom=1.15,
            prob=0.30,
            mode=["trilinear", "nearest"],  # trilinear for image, nearest for mask
            keep_size=True
        ),

        #  Step 4: Intensity Augmentation
        # Applied to IMAGE ONLY, mask stays binary 0/1

        # Gaussian noise to simulate scanner noise
        # std is in normalized [0,1] space: 0.05 ≈ 50 HU
        RandGaussianNoised(
            keys=["image"],
            mean=0.0,
            std=0.05,
            prob=0.20
        ),

        # Intensity scaling to simulate scanner calibration differences
        # factor ±10% → multiplies intensity by U(0.90, 1.10)
        RandScaleIntensityd(
            keys=["image"],
            factors=0.10,       # range: [1-0.10, 1+0.10] = [0.90, 1.10]
            prob=0.20
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=0.0,
            a_max=1.0,
            b_min=0.0,
            b_max=1.0,
            clip=True
        ),

    ])


def get_val_transforms():
    """
    Minimal pipeline for VALIDATION and TEST sets.
    NO augmentation — only load, format, and window.

    Returns:
        MONAI Compose object
    """
    return Compose([

        # Load
        LoadImaged(keys=["image", "mask"], image_only=True),
        EnsureChannelFirstd(keys=["image", "mask"]),
        EnsureTyped(keys=["image", "mask"], dtype=np.float32),

        # HU Windowing only
        ScaleIntensityRanged(
            keys=["image"],
            a_min=HU_MIN,
            a_max=HU_MAX,
            b_min=0.0,
            b_max=1.0,
            clip=True
        ),

    ])


# ── Verification ───────────────────────────────────────────────────────────────

def verify_augmentation(resampled_dir: str, n_samples: int = 2):
    """
    Spot-checks augmentation pipeline on n_samples scans.
    Verifies:
      - Image values stay in [0, 1] after augmentation
      - Mask stays strictly binary {0, 1}
      - Image and mask shapes remain identical

    Args:
        resampled_dir : path to COCA_output/data_resampled/
        n_samples     : number of scans to test (default 2)
    """
    from pathlib import Path

    resampled_path = Path(resampled_dir)
    scan_folders   = [f for f in resampled_path.iterdir() if f.is_dir()]

    if not scan_folders:
        print(f"[ERROR] No scan folders found in {resampled_dir}")
        return

    # Build sample data list
    samples = scan_folders[:n_samples]
    data_list = []
    for folder in samples:
        scan_id  = folder.name
        img_path = folder / f"{scan_id}_img.nii.gz"
        seg_path = folder / f"{scan_id}_seg.nii.gz"
        if img_path.exists() and seg_path.exists():
            data_list.append({"image": str(img_path), "mask": str(seg_path)})

    if not data_list:
        print("[ERROR] No valid image/mask pairs found.")
        return

    train_tx = get_train_transforms()

    print(f"Verifying augmentation pipeline on {len(data_list)} scans...\n")
    print(f"{'Scan':<6} {'Img Shape':<20} {'Img Min':>8} {'Img Max':>8} "
          f"{'Img Mean':>9} │ {'Mask Shape':<20} {'Mask Vals'}")
    print("─" * 90)
    all_passed = True
    for i, sample in enumerate(data_list):
        try:
            result    = train_tx(sample)
            img       = result["image"].numpy()   # (1, Z, Y, X)
            mask      = result["mask"].numpy()    # (1, Z, Y, X)
            mask_vals = np.unique(mask).tolist()

            print(f"{i:<6} {str(img.shape):<20} {img.min():>8.4f} {img.max():>8.4f} "
                  f"{img.mean():>9.4f} │ {str(mask.shape):<20} {mask_vals}")

            # Assertions
            assert img.min() >= -0.01, "Image min below 0 — clipping failed"
            assert img.max() <= 1.01,  "Image max above 1 — clipping failed"
            assert set(mask_vals).issubset({0.0, 1.0}), \
                f"Mask has non-binary values: {mask_vals}"
            assert img.shape == mask.shape, \
                f"Shape mismatch: image {img.shape} vs mask {mask.shape}"

        except Exception as e:
            print(f"{i:<6} [ERROR] {e}")
            all_passed = False
    
    print("\nChecks:")
    print(f"  All scans passed  {'YES' if all_passed else 'NO'}")


# ── Standalone run ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    RESAMPLED_DIR = r"...\COCA_output\data_resampled"
    verify_augmentation(RESAMPLED_DIR, n_samples=5)
