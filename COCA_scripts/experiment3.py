"""
experiment3.py
──────────────
Experiment 3: Full 25-scan registration and validation using
Scan 50 as atlas (best performer from Experiment 2).

Required:
  - resample_volume()            from atlas_preparation.py
  - apply_hu_window()            from atlas_preparation.py
  - register_atlas_to_scan()     from registration.py
  - apply_transform_to_volume()  from registration.py
  - REG_PARAMS                   from registration.py
  - validate_scan()              from validation.py
  - save_overlay()               from validation.py
  - save_summary_figure()        from validation.py

Outputs:
  - experiment3_output/
      ├── atlas/
      │    ├── atlas_img.nii.gz
      │    └── atlas_seg.nii.gz
      ├── registration/
      │    └── {scan_id}/
      │         ├── transform_rigid.tfm
      │         ├── transform_affine.tfm
      │         ├── warped_atlas_img.nii.gz
      │         ├── warped_atlas_seg.nii.gz
      │         └── registration_meta.json
      ├── validation/
      │    └── {scan_id}/
      │         ├── overlay_slice.png
      │         └── validation_meta.json
      ├── registration_results.csv
      ├── validation_results.csv
      └── experiment3_report.png

Usage:
    python experiment3.py
"""

import gc
import json
import time
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

#  Reuse existing modules 
import sys
sys.path.insert(0, str(Path(r"C:\Users\muham\Desktop\gsoc\code\COCA_scripts")))

from atlas_preparation import resample_volume, apply_hu_window
from registration import (
    register_atlas_to_scan,
    apply_transform_to_volume,
    REG_PARAMS
)
import validation as val_module
from validation import (
    validate_scan,
    save_overlay,
    save_summary_figure,
    DISTANCE_THRESHOLD_MM,
    TARGET_PERCENTAGE
)


#  Paths 
ATLAS_SCAN_NUM = 50   # best atlas from Experiment 2
IMAGECAS_DIR   = Path(r"...\ARCHIVE\1-200")
SPLIT_CSV      = Path(r"...\COCA_output\data_canonical\tables\split_index.csv")
RESAMPLED      = Path(r"...\COCA_output\data_resampled")
EXP3_OUT       = Path(r"...\COCA_output\experiment3_output")
ATLAS_OUT      = EXP3_OUT / "atlas"
REG_OUT        = EXP3_OUT / "registration"
VAL_OUT        = EXP3_OUT / "validation"
TARGET_SPACING = [0.7, 0.7, 3.0]


#  Step 1: Prepare Scan 50 atlas 

def prepare_atlas() -> tuple:
    """
    Loads, resamples and windows ImageCAS Scan 50.
    """
    ATLAS_OUT.mkdir(parents=True, exist_ok=True)

    img_path = IMAGECAS_DIR / f"{ATLAS_SCAN_NUM}.img.nii.gz"
    seg_path = IMAGECAS_DIR / f"{ATLAS_SCAN_NUM}.label.nii.gz"

    assert img_path.exists(), f"Not found: {img_path}"
    assert seg_path.exists(), f"Not found: {seg_path}"

    raw_img = sitk.ReadImage(str(img_path))
    raw_seg = sitk.ReadImage(str(seg_path))

    print(f"  Raw image : size={raw_img.GetSize()}  "
          f"spacing={[round(s,3) for s in raw_img.GetSpacing()]}")
    print(f"  Raw mask  : vessel_voxels="
          f"{int(sitk.GetArrayFromImage(raw_seg).sum())}")

    # Resample
    resampled_img = resample_volume(raw_img, TARGET_SPACING, is_mask=False)
    resampled_seg = resample_volume(raw_seg, TARGET_SPACING, is_mask=True)

    # HU window
    windowed_img  = apply_hu_window(resampled_img)

    # Cast to float32
    atlas_img = sitk.Cast(windowed_img,  sitk.sitkFloat32)
    atlas_seg = sitk.Cast(resampled_seg, sitk.sitkFloat32)

    # Save
    atlas_img_path = ATLAS_OUT / "atlas_img.nii.gz"
    atlas_seg_path = ATLAS_OUT / "atlas_seg.nii.gz"
    sitk.WriteImage(atlas_img, str(atlas_img_path), useCompression=True)
    sitk.WriteImage(atlas_seg, str(atlas_seg_path), useCompression=True)

    vessel_after = int(sitk.GetArrayFromImage(atlas_seg).sum())
    print(f"  Resampled : size={atlas_img.GetSize()}  "
          f"spacing={TARGET_SPACING}  vessel_voxels={vessel_after}")

    return atlas_img, atlas_seg


#  Step 2: Register all 25 candidates 

def run_registration(atlas_img: sitk.Image,
                     atlas_seg: sitk.Image,
                     candidates: pd.DataFrame) -> pd.DataFrame:
    """
    Registers Scan 50 atlas to all 25 candidate COCA scans.
    """
    REG_OUT.mkdir(parents=True, exist_ok=True)
    results = []

    print(f"\n{'─'*62}")
    print(f"{'Scan':<15} {'Category':<12} {'Rigid(s)':>9} "
          f"{'Affine(s)':>10} {'Total(s)':>9} {'Metric':>10} {'Status':>6}")
    print(f"{'─'*62}")

    for _, row in candidates.iterrows():
        gc.collect()
        scan_id  = row["scan_id"]
        category = row["category"]
        img_path = RESAMPLED / scan_id / f"{scan_id}_img.nii.gz"
        seg_path = RESAMPLED / scan_id / f"{scan_id}_seg.nii.gz"

        if not img_path.exists() or not seg_path.exists():
            print(f"{scan_id:<15} {category:<12} SKIPPED — files not found")
            continue

        scan_out = REG_OUT / scan_id
        scan_out.mkdir(parents=True, exist_ok=True)

        try:
            # Load + normalize
            fixed_img = sitk.Cast(
                sitk.ReadImage(str(img_path)), sitk.sitkFloat32
            )
            fixed_img = sitk.Clamp(fixed_img, sitk.sitkFloat32, -100.0, 900.0)
            fixed_img = sitk.ShiftScale(
                fixed_img, shift=100.0, scale=1.0 / 1000.0
            )

            t_start = time.time()

            # Retry loop
            for attempt in range(REG_PARAMS["max_retries"]):
                try:
                    rigid_tx, rigid_metric, rigid_time = \
                        register_atlas_to_scan(
                            fixed_img, atlas_img, stage="rigid"
                        )
                    resampled_atlas = sitk.Resample(
                        atlas_img, fixed_img, rigid_tx,
                        sitk.sitkLinear, 0.0, fixed_img.GetPixelID()
                    )
                    affine_tx, affine_metric, affine_time = \
                        register_atlas_to_scan(
                            fixed_img, resampled_atlas, stage="affine"
                        )
                    break
                except Exception as e:
                    if attempt < REG_PARAMS["max_retries"] - 1:
                        print(f"\n  [{scan_id}] Attempt {attempt+1} failed "
                              f"— retrying...")
                        gc.collect()
                    else:
                        raise

            total_time = time.time() - t_start

            # Compose transforms
            composite = sitk.CompositeTransform(3)
            composite.AddTransform(rigid_tx)
            composite.AddTransform(affine_tx)

            # Warp atlas image and mask 
            warped_img = apply_transform_to_volume(
                atlas_img, fixed_img, composite, is_mask=False
            )
            warped_seg = apply_transform_to_volume(
                atlas_seg, fixed_img, composite, is_mask=True
            )

            # Save transforms + warped volumes
            sitk.WriteTransform(rigid_tx,
                                str(scan_out / "transform_rigid.tfm"))
            sitk.WriteTransform(affine_tx,
                                str(scan_out / "transform_affine.tfm"))
            sitk.WriteImage(warped_img,
                            str(scan_out / "warped_atlas_img.nii.gz"),
                            useCompression=True)
            sitk.WriteImage(warped_seg,
                            str(scan_out / "warped_atlas_seg.nii.gz"),
                            useCompression=True)

            # Save metadata
            meta = {
                "scan_id"        : scan_id,
                "category"       : category,
                "atlas_scan"     : ATLAS_SCAN_NUM,
                "rigid_time_s"   : round(rigid_time, 2),
                "affine_time_s"  : round(affine_time, 2),
                "total_time_s"   : round(total_time, 2),
                "final_metric"   : round(float(affine_metric), 6),
                "rigid_metric"   : round(float(rigid_metric), 6),
                "fixed_size"     : list(fixed_img.GetSize()),
                "atlas_size"     : list(atlas_img.GetSize()),
                "reg_params"     : REG_PARAMS,
            }
            (scan_out / "registration_meta.json").write_text(
                json.dumps(meta, indent=2)
            )

            print(f"{scan_id:<15} {category:<12} {rigid_time:>9.1f} "
                  f"{affine_time:>10.1f} {total_time:>9.1f} "
                  f"{affine_metric:>10.4f} {'Successful':>6}")

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
            print(f"{scan_id:<15} {category:<12} ERROR: {str(e)[:40]}")
            results.append({
                "scan_id" : scan_id,
                "category": category,
                "status"  : f"error: {str(e)[:60]}"
            })

    results_df  = pd.DataFrame(results)
    results_csv = EXP3_OUT / "registration_results.csv"
    results_df.to_csv(results_csv, index=False)

    successful = results_df[results_df["status"] == "success"]
    print(f"\n{'='*62}")
    print(f"Registration complete:")
    print(f"  Successful  : {len(successful)} / {len(candidates)}")
    print(f"  Mean time   : {successful['total_time_s'].mean():.1f}s per scan")
    print(f"  Total time  : {successful['total_time_s'].sum()/60:.1f} minutes")
    print(f"  Mean metric : {successful['final_metric'].mean():.4f}")
    print(f" Saved → {results_csv}")

    return results_df


#  Step 3: Validate all successful registrations 

def run_validation(reg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates all successful registrations.
    Reuses validate_scan(), save_overlay(), save_summary_figure()
    from validation.py
    """
    VAL_OUT.mkdir(parents=True, exist_ok=True)

    # Patch REG_OUT in validation module to point to experiment3 registrations
    val_module.REG_OUT = REG_OUT

    successful = reg_df[reg_df["status"] == "success"].reset_index(drop=True)
    val_results = []

    print(f"\n{'─'*70}")
    print(f"{'Scan':<15} {'Cat':<10} {'Ca vox':>8} {'Within10':>9} "
          f"{'%':>7} {'MeanDist':>9} {'Pass':>6}")
    print(f"{'─'*70}")

    for _, row in successful.iterrows():
        scan_id  = row["scan_id"]
        category = row["category"]

        img_path = RESAMPLED / scan_id / f"{scan_id}_img.nii.gz"
        spacing  = list(sitk.ReadImage(str(img_path)).GetSpacing())

        # Reusing validate_scan() from validation.py
        result = validate_scan(scan_id, spacing)

        if result["status"] != "success":
            print(f"{scan_id:<15} {category:<10} "
                  f"SKIPPED: {result['status']}")
            val_results.append({
                "scan_id" : scan_id,
                "category": category,
                "status"  : result["status"]
            })
            continue

        pct    = result["percentage_10mm"]
        passed = "Passed" if result["passes_target"] else "Failed"

        print(f"{scan_id:<15} {category:<10} "
              f"{result['total_calcium']:>8} "
              f"{result['within_10mm']:>9} "
              f"{pct:>6.1f}% "
              f"{result['dist_mean_mm']:>8.1f}mm "
              f"{passed:>6}")

        # Save overlay
        scan_val_out = VAL_OUT / scan_id
        scan_val_out.mkdir(parents=True, exist_ok=True)
        save_overlay(scan_id, category, result, img_path,
                     scan_val_out / "overlay_slice.png")

        # Save metadata
        meta = {
            "scan_id"            : scan_id,
            "category"           : category,
            "atlas_scan"         : ATLAS_SCAN_NUM,
            "total_calcium"      : result["total_calcium"],
            "vessel_voxels"      : result["vessel_voxels"],
            "within_10mm"        : result["within_10mm"],
            "percentage_10mm"    : result["percentage_10mm"],
            "passes_target"      : result["passes_target"],
            "dist_mean_mm"       : result["dist_mean_mm"],
            "dist_median_mm"     : result["dist_median_mm"],
            "dist_p25_mm"        : result["dist_p25_mm"],
            "dist_p75_mm"        : result["dist_p75_mm"],
            "dist_max_mm"        : result["dist_max_mm"],
            "registration_metric": float(row["final_metric"]),
            "registration_time_s": float(row["total_time_s"]),
        }
        (scan_val_out / "validation_meta.json").write_text(
            json.dumps(meta, indent=2)
        )

        val_results.append({
            "scan_id"         : scan_id,
            "category"        : category,
            "status"          : "success",
            "total_calcium"   : result["total_calcium"],
            "vessel_voxels"   : result["vessel_voxels"],
            "within_10mm"     : result["within_10mm"],
            "percentage_10mm" : result["percentage_10mm"],
            "passes_target"   : result["passes_target"],
            "dist_mean_mm"    : result["dist_mean_mm"],
            "dist_median_mm"  : result["dist_median_mm"],
            "final_metric"    : float(row["final_metric"]),
            "total_time_s"    : float(row["total_time_s"]),
        })

    val_df  = pd.DataFrame(val_results)
    val_csv = EXP3_OUT / "validation_results.csv"
    val_df.to_csv(val_csv, index=False)

    # Summary figure
    print(f"\nGenerating summary figure...")
    success_df = val_df[val_df["status"] == "success"]
    save_summary_figure(success_df, EXP3_OUT / "experiment3_report.png")

    return val_df


# Step 4: Print final summary

def print_summary(val_df: pd.DataFrame):
    success_df = val_df[val_df["status"] == "success"]
    passed     = success_df[success_df["passes_target"] == True]

    print(f"\n{'='*62}")
    print(f"EXPERIMENT 3 — FINAL SUMMARY (Atlas: Scan {ATLAS_SCAN_NUM})")
    print(f"{'='*62}")
    print(f"  Scans validated      : {len(success_df)}")
    print(f"  Passing >70%         : {len(passed)} / {len(success_df)}")
    print(f"  Mean % within 10mm   : {success_df['percentage_10mm'].mean():.1f}%")
    print(f"  Median % within 10mm : {success_df['percentage_10mm'].median():.1f}%")
    print(f"  Min % within 10mm    : {success_df['percentage_10mm'].min():.1f}%")
    print(f"  Max % within 10mm    : {success_df['percentage_10mm'].max():.1f}%")
    print(f"  Mean dist to vessel  : {success_df['dist_mean_mm'].mean():.1f}mm")
    print(f"\n  Per-category breakdown:")
    for cat in ["Minimal", "Mild", "Moderate", "Severe"]:
        cat_df   = success_df[success_df["category"] == cat]
        if len(cat_df) == 0:
            continue
        cat_pass = (cat_df["percentage_10mm"] >= TARGET_PERCENTAGE).sum()
        print(f"    {cat:<12}: {cat_pass}/{len(cat_df)} pass  "
              f"mean={cat_df['percentage_10mm'].mean():.1f}%  "
              f"dist={cat_df['dist_mean_mm'].mean():.1f}mm")

    print(f"\n  Comparison vs Experiment 1 (Scan 100):")
    print(f"    Exp 1 mean % : 50.4%  passing: 8/25")
    print(f"    Exp 3 mean % : {success_df['percentage_10mm'].mean():.1f}%  "
          f"passing: {len(passed)}/{len(success_df)}")
    delta = success_df['percentage_10mm'].mean() - 50.4
    print(f"    Improvement  : {delta:+.1f}%")

    print(f"\nRegistration results → {EXP3_OUT}/registration_results.csv")
    print(f"   Validation results  → {EXP3_OUT}/validation_results.csv")
    print(f"   Overlays            → {EXP3_OUT}/validation/{{scan_id}}/overlay_slice.png")
    print(f"   Summary figure      → {EXP3_OUT}/experiment3_report.png")


# Main

def run_experiment3():
    print("=" * 62)
    print(f"  EXPERIMENT 3 — Full Pipeline with Atlas Scan {ATLAS_SCAN_NUM}")
    print("=" * 62)

    # Load candidates
    df         = pd.read_csv(SPLIT_CSV)
    candidates = df[df["part2_candidate"] == True].reset_index(drop=True)
    print(f"\nCandidates: {len(candidates)} scans")
    for cat, count in candidates["category"].value_counts().items():
        print(f"  {cat:<12}: {count}")

    # Step 1: Prepare atlas
    print(f"\n[Step 1] Preparing Atlas Scan {ATLAS_SCAN_NUM}...")
    atlas_img, atlas_seg = prepare_atlas()

    # Step 2: Register
    print(f"\n[Step 2] Registering atlas to {len(candidates)} scans...")
    reg_df = run_registration(atlas_img, atlas_seg, candidates)

    # Step 3: Validate
    print(f"\n[Step 3] Validating registrations...")
    val_df = run_validation(reg_df)

    # Step 4: Summary
    print_summary(val_df)


if __name__ == "__main__":
    run_experiment3()
