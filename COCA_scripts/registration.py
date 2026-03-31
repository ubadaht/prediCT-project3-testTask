"""
Step 2 of Part 2: Register ImageCAS atlas to 25 COCA non-contrast scans.

Strategy:
  - Rigid → Affine registration (no deformable — unstable for CCTA→NCCT)
  - Mattes Mutual Information metric (handles cross-modal intensity differences)
  - Multi-resolution [4x, 2x, 1x] (avoids local minima, speeds convergence)
  - Fixed random seed (reproducible sampling)
  - Retry mechanism (up to 3 attempts per scan)
  - Saves transform + warped mask per scan
  - Records timing per scan

Inputs:
  - atlas_img.nii.gz     (resampled, windowed CCTA)
  - atlas_seg.nii.gz     (resampled vessel mask)
  - split_index.csv      (25 part2_candidate scans)
  - data_resampled/      (COCA scans + calcium masks)

Outputs:
  - registration_output/
      ├── {scan_id}/
      │    ├── transform_rigid.tfm 
      │    ├── transform_affine.tfm    
      │    ├── warped_atlas_img.nii.gz   
      │    ├── warped_atlas_seg.nii.gz 
      │    └── registration_meta.json 
      └── registration_results.csv 

Usage:
    python registration.py
"""

import gc
import json
import time
import numpy as np
import pandas as pd
import SimpleITK as sitk
from pathlib import Path


# Paths
ATLAS_IMG = Path(r"...\COCA_output\atlas\atlas_img.nii.gz")
ATLAS_SEG = Path(r"...\COCA_output\atlas\atlas_seg.nii.gz")
SPLIT_CSV = Path(r"...\COCA_output\data_canonical\tables\split_index.csv")
RESAMPLED = Path(r"...\COCA_output\data_resampled")
OUT_DIR   = Path(r"...\COCA_output\registration_output")


#  Registration Parameters 
REG_PARAMS = {
    "metric"                  : "MattesMutualInformation",
    "num_histogram_bins"      : 50,
    "sampling_percentage"     : 0.10,
    "random_seed"             : 42,       # fixed seed
    "optimizer"               : "GradientDescentLineSearch",
    "learning_rate"           : 1.0,
    "num_iterations"          : 100,
    "convergence_min_value"   : 1e-6,
    "convergence_window_size" : 10,
    "shrink_factors"          : [4, 2, 1],
    "smoothing_sigmas"        : [2.0, 1.0, 0.0],
    "max_retries"             : 3,
}

def crop_to_cardiac_roi(image: sitk.Image, margin_mm: float = 80.0) -> tuple:
    """
    Crops image to a cardiac-focused region of interest.
    
    Workflow:
      XY: crop 160x160mm box around image center
          (coronary arteries are approximately centered in axial plane)
      Z:  keep middle 70% of slices
          (removes apex and base where fewer vessels exist)
    
    Args:
        image     : SimpleITK image to crop
        margin_mm : half-width of crop box in mm (default 80mm = 160mm box)
    
    Returns:
        (cropped_image, crop_params) where crop_params can be used
        to apply same crop to other images
    """
    size    = image.GetSize()       # (X, Y, Z)
    spacing = image.GetSpacing()    # (X, Y, Z)
    origin  = image.GetOrigin()

    # XY center in voxels
    cx = size[0] // 2
    cy = size[1] // 2

    # Crop half-width in voxels
    hw_x = int(margin_mm / spacing[0])
    hw_y = int(margin_mm / spacing[1])

    # Z: keep middle 70%
    z_margin = int(size[2] * 0.15)   # remove 15% from each end

    # Compute crop bounds — clamp to image bounds
    x_start = max(0, cx - hw_x)
    x_end   = min(size[0], cx + hw_x)
    y_start = max(0, cy - hw_y)
    y_end   = min(size[1], cy + hw_y)
    z_start = max(0, z_margin)
    z_end   = min(size[2], size[2] - z_margin)

    crop_params = {
        "x_start": x_start, "x_end": x_end,
        "y_start": y_start, "y_end": y_end,
        "z_start": z_start, "z_end": z_end
    }

    # SimpleITK crop using RegionOfInterest
    roi = sitk.RegionOfInterestImageFilter()
    roi.SetSize([x_end - x_start, y_end - y_start, z_end - z_start])
    roi.SetIndex([x_start, y_start, z_start])

    return roi.Execute(image), crop_params


def apply_crop(image: sitk.Image, crop_params: dict) -> sitk.Image:
    """Applies pre-computed crop params to another image (e.g. atlas)."""
    p   = crop_params
    roi = sitk.RegionOfInterestImageFilter()
    roi.SetSize([
        p["x_end"] - p["x_start"],
        p["y_end"] - p["y_start"],
        p["z_end"] - p["z_start"]
    ])
    roi.SetIndex([p["x_start"], p["y_start"], p["z_start"]])
    return roi.Execute(image)
#  Registration Function 

def register_atlas_to_scan(
    fixed_img  : sitk.Image,
    moving_img : sitk.Image,
    stage      : str = "rigid"
) -> tuple:
    """
    Registers moving image (atlas) to fixed image (COCA scan).

    Two-stage registration:
      Stage 1 — Rigid  (translation + rotation, 6 DOF)
      Stage 2 — Affine (+ scaling + shear, 12 DOF)

    Args:
        fixed_img  : COCA scan (target space)
        moving_img : Atlas (source space)
        stage      : "rigid" or "affine"

    Returns:
        (transform, final_metric_value, elapsed_seconds)
    """
    registration = sitk.ImageRegistrationMethod()

    #  MI
    registration.SetMetricAsMattesMutualInformation(
        numberOfHistogramBins=REG_PARAMS["num_histogram_bins"]
    )
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(
    REG_PARAMS["sampling_percentage"],
    seed=REG_PARAMS["random_seed"]
    )

    #  Interpolator 
    registration.SetInterpolator(sitk.sitkLinear)

    #  Optimizer 
    # GradientDescentLineSearch
    registration.SetOptimizerAsGradientDescentLineSearch(
        learningRate=REG_PARAMS["learning_rate"],
        numberOfIterations=REG_PARAMS["num_iterations"],
        convergenceMinimumValue=REG_PARAMS["convergence_min_value"],
        convergenceWindowSize=REG_PARAMS["convergence_window_size"]
    )
    registration.SetOptimizerScalesFromPhysicalShift()

    #  Multi-resolution pyramid 
    # [4, 2, 1] coarse to fine alignment
    registration.SetShrinkFactorsPerLevel(
        shrinkFactors=REG_PARAMS["shrink_factors"]
    )
    registration.SetSmoothingSigmasPerLevel(
        smoothingSigmas=REG_PARAMS["smoothing_sigmas"]
    )
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    #  Transform initialization 
    if stage == "rigid":
        initial_transform = sitk.CenteredTransformInitializer(
            fixed_img,
            moving_img,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.MOMENTS
        )
    else:  # affine
        initial_transform = sitk.CenteredTransformInitializer(
            fixed_img,
            moving_img,
            sitk.AffineTransform(3),
            sitk.CenteredTransformInitializerFilter.MOMENTS
        )

    registration.SetInitialTransform(initial_transform, inPlace=False)

    #  Execute 
    t0        = time.time()
    transform = registration.Execute(fixed_img, moving_img)
    elapsed   = time.time() - t0

    return transform, registration.GetMetricValue(), elapsed


def apply_transform_to_volume(
    moving    : sitk.Image,
    fixed     : sitk.Image,
    transform : sitk.Transform,
    is_mask   : bool = False
) -> sitk.Image:
    """
    Warps moving image into fixed image space using computed transform.

    Args:
        moving    : image to warp (atlas)
        fixed     : reference space (COCA scan)
        transform : registration transform
        is_mask   : True → NearestNeighbor, False → Linear

    Returns:
        warped SimpleITK image
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetTransform(transform)
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(
        sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear
    )
    return resampler.Execute(moving)


#  Main Pipeline 

def run_registration_pipeline():
    print("=" * 60)
    print("  ATLAS REGISTRATION PIPELINE — Step 2 of Part 2")
    print("=" * 60)

    # Load atlas 
    print("\nLoading atlas...")
    assert ATLAS_IMG.exists(), f"Atlas image not found: {ATLAS_IMG}"
    assert ATLAS_SEG.exists(), f"Atlas mask not found:  {ATLAS_SEG}"

    atlas_img = sitk.Cast(sitk.ReadImage(str(ATLAS_IMG)), sitk.sitkFloat32)
    atlas_seg = sitk.Cast(sitk.ReadImage(str(ATLAS_SEG)), sitk.sitkFloat32)
    print(f"  Atlas image : {atlas_img.GetSize()}  "
          f"spacing={[round(s,2) for s in atlas_img.GetSpacing()]}")
    print(f"  Atlas mask  : vessel voxels = "
          f"{int(sitk.GetArrayFromImage(atlas_seg).sum())}")

    # ── Load candidates ────────────────────────────────────────────────────
    print("\nLoading Part 2 candidate scans from split_index.csv...")
    df         = pd.read_csv(SPLIT_CSV)
    candidates = df[df["part2_candidate"] == True].reset_index(drop=True)
    print(f"  Found {len(candidates)} candidate scans")
    print(f"  Category breakdown:")
    for cat, count in candidates["category"].value_counts().items():
        print(f"    {cat:<12}: {count}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Registration loop 
    results = []
    print(f"\n{'─'*60}")
    print(f"{'Scan':<15} {'Category':<12} {'Rigid(s)':>9} {'Affine(s)':>10} "
          f"{'Total(s)':>9} {'Metric':>10} {'Status':>8}")
    print(f"{'─'*60}")

    for idx, row in candidates.iterrows():
        gc.collect()
        scan_id  = row["scan_id"]
        category = row["category"]
        img_path = RESAMPLED / scan_id / f"{scan_id}_img.nii.gz"
        seg_path = RESAMPLED / scan_id / f"{scan_id}_seg.nii.gz"

        if not img_path.exists() or not seg_path.exists():
            print(f"{scan_id:<15} {category:<12} "
                  f"{'SKIPPED — files not found':>40}")
            continue

        scan_out = OUT_DIR / scan_id
        scan_out.mkdir(parents=True, exist_ok=True)

        try:
            # Load + normalize fixed image 
            fixed_img = sitk.Cast(
                sitk.ReadImage(str(img_path)), sitk.sitkFloat32
            )
            fixed_seg = sitk.ReadImage(str(seg_path))

            # HU window to match atlas intensity space
            fixed_img = sitk.Clamp(fixed_img, sitk.sitkFloat32, -100.0, 900.0)
            fixed_img = sitk.ShiftScale(
                fixed_img, shift=100.0, scale=1.0 / 1000.0
            )

            t_total_start = time.time()

            #  Crop both images to cardiac ROI for registration 
            fixed_cropped, crop_params = crop_to_cardiac_roi(fixed_img)

            # Atlas must be same size as fixed for crop, resample atlas
            # to fixed image space first, then crop
            atlas_resampled_to_fixed = sitk.Resample(
            atlas_img, fixed_img,
            sitk.Transform(),       # identity: just resample to same grid
            sitk.sitkLinear, 0.0,
            fixed_img.GetPixelID()
            )
            atlas_cropped = apply_crop(atlas_resampled_to_fixed, crop_params)

            #  Retry loop 
            rigid_transform = affine_transform = None
            rigid_metric = affine_metric = rigid_time = affine_time = 0.0

            for attempt in range(REG_PARAMS["max_retries"]):
                try:
                    # Stage 1: Rigid
                    rigid_transform, rigid_metric, rigid_time = \
                        register_atlas_to_scan(
                            fixed_img, atlas_img, stage="rigid"
                        )

                    # Resample atlas with rigid result, match pixel type
                    resampled_atlas = sitk.Resample(
                        atlas_img,
                        fixed_img,
                        rigid_transform,
                        sitk.sitkLinear,
                        0.0,
                        fixed_img.GetPixelID()
                    )

                    # Stage 2: Affine initialized from rigid-aligned atlas
                    affine_transform, affine_metric, affine_time = \
                        register_atlas_to_scan(
                            fixed_img, resampled_atlas, stage="affine"
                        )

                    break  # success exit retry loop

                except Exception as retry_err:
                    if attempt < REG_PARAMS["max_retries"] - 1:
                        print(f"\n  [{scan_id}] Attempt {attempt + 1} failed "
                              f"— retrying... ({retry_err})")
                        gc.collect()
                    else:
                        raise  # all retries exhausted — propagate to outer except

            total_time = time.time() - t_total_start

            #  Compose rigid + affine into single transform 
            composite = sitk.CompositeTransform(3)
            composite.AddTransform(rigid_transform)
            composite.AddTransform(affine_transform)

            #  Apply composite transform to atlas image and mask 
            warped_img = apply_transform_to_volume(
                atlas_img, fixed_img, composite, is_mask=False
            )
            warped_seg = apply_transform_to_volume(
                atlas_seg, fixed_img, composite, is_mask=True
            )

            # Save transforms 
            sitk.WriteTransform(rigid_transform,
                                str(scan_out / "transform_rigid.tfm"))
            sitk.WriteTransform(affine_transform,
                                str(scan_out / "transform_affine.tfm"))

            # Save warped volumes 
            sitk.WriteImage(warped_img,
                            str(scan_out / "warped_atlas_img.nii.gz"),
                            useCompression=True)
            sitk.WriteImage(warped_seg,
                            str(scan_out / "warped_atlas_seg.nii.gz"),
                            useCompression=True)

            #  Save per-scan metadata 
            meta = {
                "scan_id"            : scan_id,
                "category"           : category,
                "rigid_time_s"       : round(rigid_time, 2),
                "affine_time_s"      : round(affine_time, 2),
                "total_time_s"       : round(total_time, 2),
                "final_metric"       : round(float(affine_metric), 6),
                "rigid_metric"       : round(float(rigid_metric), 6),
                "fixed_size"         : list(fixed_img.GetSize()),
                "atlas_size"         : list(atlas_img.GetSize()),
                "registration_params": REG_PARAMS
            }
            (scan_out / "registration_meta.json").write_text(
                json.dumps(meta, indent=2)
            )

            print(f"{scan_id:<15} {category:<12} {rigid_time:>9.1f} "
                  f"{affine_time:>10.1f} {total_time:>9.1f} "
                  f"{affine_metric:>10.4f} {'Successful':>8}")

            results.append({
                "scan_id"      : scan_id,
                "category"     : category,
                "rigid_time_s" : round(rigid_time, 2),
                "affine_time_s": round(affine_time, 2),
                "total_time_s" : round(total_time, 2),
                "final_metric" : round(float(affine_metric), 6),
                "status"       : "success"
            })

            gc.collect()
            sitk.ProcessObject_GlobalWarningDisplayOff()

        except Exception as e:
            print(f"{scan_id:<15} {category:<12} "
                  f"{'ERROR: ' + str(e)[:35]:>40}")
            results.append({
                "scan_id" : scan_id,
                "category": category,
                "status"  : f"error: {str(e)[:50]}"
            })

    #  Save results CSV 
    results_df  = pd.DataFrame(results)
    results_csv = OUT_DIR / "registration_results.csv"
    results_df.to_csv(results_csv, index=False)

    #  Summary 
    print(f"\n{'='*60}")
    successful = results_df[results_df["status"] == "success"]
    if len(successful) > 0:
        print(f"Registration complete:")
        print(f"  Successful   : {len(successful)} / {len(candidates)}")
        print(f"  Mean time    : {successful['total_time_s'].mean():.1f}s per scan")
        print(f"  Median time  : {successful['total_time_s'].median():.1f}s per scan")
        print(f"  Total time   : {successful['total_time_s'].sum()/60:.1f} minutes")
        print(f"  Mean metric  : {successful['final_metric'].mean():.4f}")
    print(f"\n Results saved → {results_csv}")
    print(f"   Output dir   → {OUT_DIR}")


if __name__ == "__main__":
    run_registration_pipeline()
