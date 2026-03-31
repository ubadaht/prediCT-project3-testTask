"""
Usage:
    python atlas_preparation.py
"""

import numpy as np
import SimpleITK as sitk
from pathlib import Path


# Paths
ATLAS_IMG_PATH = Path(r"...ARCHIVE\1-200\100.img.nii.gz")
ATLAS_SEG_PATH = Path(r"...ARCHIVE\1-200\100.label.nii.gz")
OUT_DIR        = Path(r"...\COCA_output\atlas")

# Target spacing — must match COCA resampled scans
TARGET_SPACING = [0.7, 0.7, 3.0]   # recommended values

# HU Window
HU_MIN = -100.0
HU_MAX =  900.0


def resample_volume(volume: sitk.Image,
                    target_spacing: list,
                    is_mask: bool = False) -> sitk.Image:
    """
    Resamples a SimpleITK image to target_spacing.

    Args:
        volume         : input SimpleITK image
        target_spacing : [X, Y, Z] spacing in mm
        is_mask        : True → NearestNeighbor (preserve binary labels)
                         False → Linear (smooth HU interpolation)

    Returns:
        resampled SimpleITK image
    """
    original_spacing = volume.GetSpacing()    # (X, Y, Z)
    original_size    = volume.GetSize()       # (X, Y, Z)

    # NewSize = OldSize * (OldSpacing / NewSpacing)
    new_size = [
        int(round(original_size[i] * (original_spacing[i] / target_spacing[i])))
        for i in range(3)
    ]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(target_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(volume.GetDirection())
    resample.SetOutputOrigin(volume.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(0)

    if is_mask:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkLinear)

    return resample.Execute(volume)


def apply_hu_window(image: sitk.Image,
                    hu_min: float = HU_MIN,
                    hu_max: float = HU_MAX) -> sitk.Image:
    """
    Clips HU values to [hu_min, hu_max] and normalizes to [0, 1].
    Applied to image only — not mask.

    Args:
        image  : SimpleITK image with raw HU values
        hu_min : lower HU bound (default -100)
        hu_max : upper HU bound (default  900)

    Returns:
        SimpleITK image with float32 values in [0, 1]
    """
    image = sitk.Cast(image, sitk.sitkFloat32)

    # Clip to window
    image = sitk.Clamp(image, sitk.sitkFloat32, hu_min, hu_max)

    # Normalize to [0, 1]
    # (v - hu_min) / (hu_max - hu_min)
    image = sitk.ShiftScale(
        image,
        shift=-hu_min,
        scale=1.0 / (hu_max - hu_min)
    )

    return image


def print_volume_stats(label: str, volume: sitk.Image, is_mask: bool = False):
    """Prints key statistics for a SimpleITK volume."""
    arr = sitk.GetArrayFromImage(volume).astype(np.float32)
    print(f"\n  {label}:")
    print(f"    Size    : {volume.GetSize()}")
    print(f"    Spacing : {[round(s, 3) for s in volume.GetSpacing()]} mm")
    print(f"    Origin  : {[round(o, 1) for o in volume.GetOrigin()]}")
    if is_mask:
        print(f"    Unique  : {np.unique(arr).tolist()}")
        print(f"    Vessel voxels: {int(arr.sum())}")
    else:
        print(f"    HU min  : {arr.min():.2f}")
        print(f"    HU max  : {arr.max():.2f}")
        print(f"    HU mean : {arr.mean():.2f}")
        print(f"    HU std  : {arr.std():.2f}")


# Main

if __name__ == "__main__":

    assert ATLAS_IMG_PATH.exists(), f"Atlas image not found: {ATLAS_IMG_PATH}"
    assert ATLAS_SEG_PATH.exists(), f"Atlas mask not found:  {ATLAS_SEG_PATH}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("  ATLAS PREPARATION — Step 1 of Part 2")
    print("=" * 55)

    print("\nLoading atlas files...")
    atlas_img = sitk.ReadImage(str(ATLAS_IMG_PATH))
    atlas_seg = sitk.ReadImage(str(ATLAS_SEG_PATH))

    print("\nBEFORE resampling:")
    print_volume_stats("Atlas image", atlas_img, is_mask=False)
    print_volume_stats("Atlas mask",  atlas_seg, is_mask=True)

    # Resample 
    print(f"\nResampling to {TARGET_SPACING} mm...")
    resampled_img = resample_volume(atlas_img, TARGET_SPACING, is_mask=False)
    resampled_seg = resample_volume(atlas_seg, TARGET_SPACING, is_mask=True)

    print("\nAFTER resampling:")
    print_volume_stats("Atlas image (resampled)", resampled_img, is_mask=False)
    print_volume_stats("Atlas mask  (resampled)", resampled_seg, is_mask=True)

    # Apply HU windowing to image
    print("\nApplying HU windowing [-100, 900] → [0, 1]...")
    windowed_img = apply_hu_window(resampled_img)
    print_volume_stats("Atlas image (windowed)", windowed_img, is_mask=False)

    # Save
    print("\nSaving atlas files...")

    atlas_img_out = OUT_DIR / "atlas_img.nii.gz"
    atlas_seg_out = OUT_DIR / "atlas_seg.nii.gz"

    sitk.WriteImage(windowed_img, str(atlas_img_out), useCompression=True)
    sitk.WriteImage(resampled_seg, str(atlas_seg_out), useCompression=True)

    print(f"\n Atlas prepared successfully:")
    print(f"   Image → {atlas_img_out}")
    print(f"   Mask  → {atlas_seg_out}")

    # Sanity check
    print("\nSanity checks:")
    arr_img = sitk.GetArrayFromImage(windowed_img)
    arr_seg = sitk.GetArrayFromImage(resampled_seg)

    img_ok  = arr_img.min() >= -0.01 and arr_img.max() <= 1.01
    mask_ok = set(np.unique(arr_seg)).issubset({0, 1, 0.0, 1.0})
    space_ok = list(windowed_img.GetSpacing()) == TARGET_SPACING

    print(f"  Image range [0,1]       : {'✅' if img_ok   else '❌'} "
          f"[{arr_img.min():.4f}, {arr_img.max():.4f}]")
    print(f"  Mask binary {{0,1}}       : {'✅' if mask_ok  else '❌'} "
          f"{np.unique(arr_seg).tolist()}")
    print(f"  Spacing = {TARGET_SPACING} : {'✅' if space_ok else '❌'} "
          f"{list(windowed_img.GetSpacing())}")
    print(f"  Vessel voxels preserved : {int(arr_seg.sum())} "
          f"(was {int(sitk.GetArrayFromImage(atlas_seg).sum())})")
