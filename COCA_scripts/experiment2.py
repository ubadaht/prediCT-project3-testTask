"""
experiment2.py
──────────────
Experiment 2: Atlas candidate comparison.

Tests 4 atlas candidates (Scan 1, 10, 50, 100) against a
representative subset of 8 COCA scans (2 per CAC category).

Goal: Find which atlas gives best validation score and understand why.

Reuses:
  - resample_volume()     from atlas_preparation.py
  - apply_hu_window()     from atlas_preparation.py
  - register_atlas_to_scan()     from registration.py
  - apply_transform_to_volume()  from registration.py
  - validate_scan()              from validation.py

Outputs:
  - experiment2_output/
      ├── atlas_{scan_num}/
      │    ├── atlas_img.nii.gz        ← resampled candidate
      │    ├── atlas_seg.nii.gz        ← resampled vessel mask
      │    └── atlas_stats.json        ← size, spacing, vessel voxels
      ├── results/
      │    ├── {atlas}_{scan_id}_warped_seg.nii.gz
      │    └── comparison_table.csv
      └── experiment2_report.png       ← comparison figure

Usage:
    python experiment2.py
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
from scipy.ndimage import distance_transform_edt

# ── Reuse existing modules ─────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(r"C:\Users\muham\Desktop\gsoc\code\COCA_scripts")))

from atlas_preparation import resample_volume, apply_hu_window
from registration import register_atlas_to_scan, apply_transform_to_volume, REG_PARAMS
from validation import validate_scan, DISTANCE_THRESHOLD_MM, TARGET_PERCENTAGE


# ── Paths ──────────────────────────────────────────────────────────────────────
IMAGECAS_DIR = Path(r"...\ARCHIVE\1-200")
SPLIT_CSV    = Path(r"...\COCA_output\data_canonical\tables\split_index.csv")
RESAMPLED    = Path(r"...\COCA_output\data_resampled")
EXP2_OUT     = Path(r"...\COCA_output\experiment2_output")
REG_OUT_BASE = EXP2_OUT / "registrations"
TARGET_SPACING = [0.7, 0.7, 3.0]

# ── Atlas candidates to test ───────────────────────────────────────────────────
# Scan 100 is our current baseline — included for fair comparison
ATLAS_CANDIDATES = [1, 10, 50, 100]

# ── Subset: 2 scans per category for fast comparison ──────────────────────────
# Fixed subset — same 8 scans tested against all 4 atlases
SUBSET_PER_CATEGORY = 2


# ── Step 1: Prepare atlas candidate ───────────────────────────────────────────

def prepare_atlas_candidate(scan_num: int, out_dir: Path) -> dict:
    """
    Loads, resamples, and windows an ImageCAS atlas candidate.
    Reuses resample_volume() and apply_hu_window() from atlas_preparation.py

    Args:
        scan_num : ImageCAS scan number (1, 10, 50, 100)
        out_dir  : output directory for this candidate

    Returns:
        dict with atlas_img, atlas_seg, stats
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    img_path = IMAGECAS_DIR / f"{scan_num}.img.nii.gz"
    seg_path = IMAGECAS_DIR / f"{scan_num}.label.nii.gz"

    assert img_path.exists(), f"Atlas image not found: {img_path}"
    assert seg_path.exists(), f"Atlas mask not found:  {seg_path}"

    # Load
    raw_img = sitk.ReadImage(str(img_path))
    raw_seg = sitk.ReadImage(str(seg_path))

    # Resample — reusing function from atlas_preparation.py
    resampled_img = resample_volume(raw_img, TARGET_SPACING, is_mask=False)
    resampled_seg = resample_volume(raw_seg, TARGET_SPACING, is_mask=True)

    # HU window → [0,1] — reusing function from atlas_preparation.py
    windowed_img = apply_hu_window(resampled_img)

    # Cast
    windowed_img = sitk.Cast(windowed_img, sitk.sitkFloat32)
    resampled_seg = sitk.Cast(resampled_seg, sitk.sitkFloat32)

    # Save
    atlas_img_path = out_dir / "atlas_img.nii.gz"
    atlas_seg_path = out_dir / "atlas_seg.nii.gz"
    sitk.WriteImage(windowed_img,  str(atlas_img_path), useCompression=True)
    sitk.WriteImage(resampled_seg, str(atlas_seg_path), useCompression=True)

    # Stats
    arr_img = sitk.GetArrayFromImage(windowed_img)
    arr_seg = sitk.GetArrayFromImage(resampled_seg)
    vessel_voxels_before = int(sitk.GetArrayFromImage(raw_seg).sum())
    vessel_voxels_after  = int(arr_seg.sum())

    stats = {
        "scan_num"             : scan_num,
        "original_size"        : list(raw_img.GetSize()),
        "original_spacing"     : [round(s, 3) for s in raw_img.GetSpacing()],
        "resampled_size"       : list(windowed_img.GetSize()),
        "resampled_spacing"    : TARGET_SPACING,
        "vessel_voxels_before" : vessel_voxels_before,
        "vessel_voxels_after"  : vessel_voxels_after,
        "img_mean"             : round(float(arr_img.mean()), 4),
        "img_std"              : round(float(arr_img.std()), 4),
    }
    (out_dir / "atlas_stats.json").write_text(json.dumps(stats, indent=2))

    print(f"  Scan {scan_num:>3}: size={list(windowed_img.GetSize())}  "
          f"spacing={TARGET_SPACING}  "
          f"vessel_voxels={vessel_voxels_after}  "
          f"(was {vessel_voxels_before})")

    return {
        "scan_num"   : scan_num,
        "atlas_img"  : windowed_img,
        "atlas_seg"  : resampled_seg,
        "stats"      : stats,
        "img_path"   : atlas_img_path,
        "seg_path"   : atlas_seg_path,
    }


# ── Step 2: Select evaluation subset ──────────────────────────────────────────

def select_eval_subset(n_per_category: int = 2) -> pd.DataFrame:
    """
    Selects a balanced subset of part2_candidate scans for atlas comparison.
    Picks n_per_category scans from each CAC category.

    Args:
        n_per_category : scans per category (default 2)

    Returns:
        DataFrame with selected scans
    """
    df         = pd.read_csv(SPLIT_CSV)
    candidates = df[df["part2_candidate"] == True]

    subset_rows = []
    for cat in ["Minimal", "Mild", "Moderate", "Severe"]:
        cat_df = candidates[candidates["category"] == cat]
        # Pick scans with most calcium voxels — more informative validation
        picked = cat_df.nlargest(n_per_category, "voxels")
        subset_rows.append(picked)

    subset = pd.concat(subset_rows).reset_index(drop=True)
    return subset


# ── Step 3: Register one atlas to one scan ────────────────────────────────────

def register_and_validate(
    atlas_img   : sitk.Image,
    atlas_seg   : sitk.Image,
    scan_id     : str,
    category    : str,
    out_dir     : Path
) -> dict:
    """
    Registers atlas to one COCA scan and validates.
    Reuses register_atlas_to_scan() and apply_transform_to_volume()
    from registration.py, and validate_scan() from validation.py

    Returns:
        dict with registration + validation metrics
    """
    img_path = RESAMPLED / scan_id / f"{scan_id}_img.nii.gz"
    seg_path = RESAMPLED / scan_id / f"{scan_id}_seg.nii.gz"

    if not img_path.exists():
        return {"status": "missing_file"}

    # Load + normalize fixed image
    fixed_img = sitk.Cast(sitk.ReadImage(str(img_path)), sitk.sitkFloat32)
    fixed_img = sitk.Clamp(fixed_img, sitk.sitkFloat32, -100.0, 900.0)
    fixed_img = sitk.ShiftScale(fixed_img, shift=100.0, scale=1.0 / 1000.0)

    t0 = time.time()

    # Registration with retry — reusing register_atlas_to_scan()
    for attempt in range(REG_PARAMS["max_retries"]):
        try:
            rigid_tx, rigid_metric, rigid_time = register_atlas_to_scan(
                fixed_img, atlas_img, stage="rigid"
            )
            resampled_atlas = sitk.Resample(
                atlas_img, fixed_img, rigid_tx,
                sitk.sitkLinear, 0.0, fixed_img.GetPixelID()
            )
            affine_tx, affine_metric, affine_time = register_atlas_to_scan(
                fixed_img, resampled_atlas, stage="affine"
            )
            break
        except Exception as e:
            if attempt < REG_PARAMS["max_retries"] - 1:
                gc.collect()
            else:
                return {"status": f"registration_failed: {str(e)[:60]}"}

    total_time = time.time() - t0

    # Compose + warp vessel mask
    composite = sitk.CompositeTransform(3)
    composite.AddTransform(rigid_tx)
    composite.AddTransform(affine_tx)

    warped_seg = apply_transform_to_volume(
        atlas_seg, fixed_img, composite, is_mask=True
    )

    # Save warped seg for validation
    out_dir.mkdir(parents=True, exist_ok=True)
    warped_path = out_dir / f"{scan_id}_warped_seg.nii.gz"
    sitk.WriteImage(warped_seg, str(warped_path), useCompression=True)

    # Validate — reusing validate_scan() from validation.py
    # Temporarily write warped seg to expected path
    temp_reg_dir = EXP2_OUT / "temp_reg" / scan_id
    temp_reg_dir.mkdir(parents=True, exist_ok=True)
    temp_warped = temp_reg_dir / "warped_atlas_seg.nii.gz"
    sitk.WriteImage(warped_seg, str(temp_warped), useCompression=True)

    # Patch REG_OUT for validate_scan
    import validation as val_module
    original_reg_out = val_module.REG_OUT
    val_module.REG_OUT = EXP2_OUT / "temp_reg"

    spacing = list(sitk.ReadImage(str(img_path)).GetSpacing())
    val_result = validate_scan(scan_id, spacing)

    # Restore
    val_module.REG_OUT = original_reg_out

    if val_result["status"] != "success":
        return {"status": val_result["status"]}

    gc.collect()

    return {
        "status"          : "success",
        "scan_id"         : scan_id,
        "category"        : category,
        "rigid_time_s"    : round(rigid_time, 2),
        "affine_time_s"   : round(affine_time, 2),
        "total_time_s"    : round(total_time, 2),
        "final_metric"    : round(float(affine_metric), 6),
        "total_calcium"   : val_result["total_calcium"],
        "vessel_voxels"   : val_result["vessel_voxels"],
        "within_10mm"     : val_result["within_10mm"],
        "percentage_10mm" : val_result["percentage_10mm"],
        "passes_target"   : val_result["passes_target"],
        "dist_mean_mm"    : val_result["dist_mean_mm"],
        "dist_median_mm"  : val_result["dist_median_mm"],
    }


# ── Step 4: Comparison figure ──────────────────────────────────────────────────

def save_comparison_figure(all_results: dict, out_path: Path):
    """
    4-panel comparison figure:
      1. Mean % within 10mm per atlas
      2. Per-category breakdown per atlas
      3. Registration time per atlas
      4. MI metric per atlas
    """
    CAT_COLORS  = {
        "Minimal": "#60a5fa", "Mild": "#34d399",
        "Moderate": "#f59e0b", "Severe": "#f87171"
    }
    ATLAS_COLORS = ["#6366f1", "#22d3ee", "#f472b6", "#fb923c"]
    categories   = ["Minimal", "Mild", "Moderate", "Severe"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.patch.set_facecolor("white")
    fig.suptitle("Experiment 2: Atlas Candidate Comparison",
                 fontsize=13, fontweight="bold")

    atlas_labels = [f"Scan {n}" for n in ATLAS_CANDIDATES]
    means        = []
    pass_rates   = []
    mean_times   = []
    mean_metrics = []

    for scan_num in ATLAS_CANDIDATES:
        df = all_results[scan_num]
        ok = df[df["status"] == "success"]
        means.append(ok["percentage_10mm"].mean() if len(ok) > 0 else 0)
        pass_rates.append(
            100 * (ok["percentage_10mm"] >= TARGET_PERCENTAGE).mean()
            if len(ok) > 0 else 0
        )
        mean_times.append(ok["total_time_s"].mean() if len(ok) > 0 else 0)
        mean_metrics.append(ok["final_metric"].abs().mean() if len(ok) > 0 else 0)

    # ── Panel 1: Mean % within 10mm ───────────────────────────────────────
    ax = axes[0, 0]
    bars = ax.bar(atlas_labels, means, color=ATLAS_COLORS,
                  edgecolor="white", width=0.5)
    ax.axhline(TARGET_PERCENTAGE, color="#dc2626", linewidth=1.5,
               linestyle="--", label=f"Target ({TARGET_PERCENTAGE:.0f}%)")
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom",
                fontsize=9, fontweight="bold")
    ax.set_ylabel("Mean % calcium within ±10mm")
    ax.set_title("Mean Validation Score per Atlas", fontweight="bold")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # ── Panel 2: Per-category breakdown ───────────────────────────────────
    ax = axes[0, 1]
    x      = np.arange(len(categories))
    width  = 0.2
    for i, (scan_num, color) in enumerate(zip(ATLAS_CANDIDATES, ATLAS_COLORS)):
        df   = all_results[scan_num]
        ok   = df[df["status"] == "success"]
        vals = [
            ok[ok["category"] == cat]["percentage_10mm"].mean()
            if len(ok[ok["category"] == cat]) > 0 else 0
            for cat in categories
        ]
        ax.bar(x + i * width, vals, width, label=f"Scan {scan_num}",
               color=color, edgecolor="white")
    ax.axhline(TARGET_PERCENTAGE, color="#dc2626", linewidth=1,
               linestyle="--", alpha=0.7)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Mean % within ±10mm")
    ax.set_title("Per-Category Breakdown", fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 115)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # ── Panel 3: Registration time ─────────────────────────────────────────
    ax = axes[1, 0]
    bars = ax.bar(atlas_labels, mean_times, color=ATLAS_COLORS,
                  edgecolor="white", width=0.5)
    for bar, val in zip(bars, mean_times):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f"{val:.1f}s", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Mean registration time (s)")
    ax.set_title("Registration Speed per Atlas", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # ── Panel 4: MI metric ────────────────────────────────────────────────
    ax = axes[1, 1]
    bars = ax.bar(atlas_labels, mean_metrics, color=ATLAS_COLORS,
                  edgecolor="white", width=0.5)
    for bar, val in zip(bars, mean_metrics):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("|MI metric| (higher = better)")
    ax.set_title("Registration Quality (MI Metric)", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(str(out_path), dpi=130, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"  Comparison figure → {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def run_experiment2():
    print("=" * 65)
    print("  EXPERIMENT 2 — Atlas Candidate Comparison")
    print("=" * 65)

    EXP2_OUT.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Prepare all atlas candidates ──────────────────────────────
    print("\n[Step 1] Preparing atlas candidates...")
    atlas_data = {}
    for scan_num in ATLAS_CANDIDATES:
        out_dir = EXP2_OUT / f"atlas_{scan_num}"
        print(f"\n  Atlas Scan {scan_num}:")
        atlas_data[scan_num] = prepare_atlas_candidate(scan_num, out_dir)

    # ── Step 2: Select evaluation subset ──────────────────────────────────
    print(f"\n[Step 2] Selecting evaluation subset "
          f"({SUBSET_PER_CATEGORY} per category)...")
    subset = select_eval_subset(SUBSET_PER_CATEGORY)
    print(f"  Selected {len(subset)} scans:")
    for _, row in subset.iterrows():
        print(f"    {row['scan_id']}  {row['category']:<12}  "
              f"voxels={row['voxels']}")

    # ── Step 3: Register + validate each atlas on subset ──────────────────
    print(f"\n[Step 3] Registering and validating...")
    all_results = {}

    for scan_num in ATLAS_CANDIDATES:
        print(f"\n  {'─'*55}")
        print(f"  Atlas Scan {scan_num}  "
              f"(vessel_voxels="
              f"{atlas_data[scan_num]['stats']['vessel_voxels_after']})")
        print(f"  {'─'*55}")
        print(f"  {'Scan':<15} {'Cat':<10} {'Time':>7} "
              f"{'Metric':>8} {'%10mm':>7} {'Pass':>6}")
        print(f"  {'─'*55}")

        atlas_img = atlas_data[scan_num]["atlas_img"]
        atlas_seg = atlas_data[scan_num]["atlas_seg"]
        results   = []

        for _, row in subset.iterrows():
            scan_id  = row["scan_id"]
            category = row["category"]
            out_dir  = EXP2_OUT / "registrations" / f"atlas_{scan_num}" / scan_id

            result = register_and_validate(
                atlas_img, atlas_seg, scan_id, category, out_dir
            )

            if result["status"] == "success":
                passed = "✅" if result["passes_target"] else "❌"
                print(f"  {scan_id:<15} {category:<10} "
                      f"{result['total_time_s']:>6.1f}s "
                      f"{result['final_metric']:>8.4f} "
                      f"{result['percentage_10mm']:>6.1f}% "
                      f"{passed:>6}")
            else:
                print(f"  {scan_id:<15} {category:<10} "
                      f"{'FAILED: ' + result['status'][:30]:>45}")

            result["scan_id"]  = scan_id
            result["category"] = category
            results.append(result)
            gc.collect()

        results_df = pd.DataFrame(results)
        all_results[scan_num] = results_df

        ok = results_df[results_df["status"] == "success"]
        if len(ok) > 0:
            print(f"\n  Summary — Atlas Scan {scan_num}:")
            print(f"    Mean % within 10mm : {ok['percentage_10mm'].mean():.1f}%")
            print(f"    Passing (>70%)     : "
                  f"{(ok['percentage_10mm'] >= TARGET_PERCENTAGE).sum()}"
                  f"/{len(ok)}")
            print(f"    Mean time          : {ok['total_time_s'].mean():.1f}s")
            print(f"    Mean |MI metric|   : "
                  f"{ok['final_metric'].abs().mean():.4f}")

    # ── Step 4: Save comparison table ─────────────────────────────────────
    print(f"\n[Step 4] Saving results...")
    comparison_rows = []
    for scan_num in ATLAS_CANDIDATES:
        ok = all_results[scan_num]
        ok = ok[ok["status"] == "success"]
        if len(ok) == 0:
            continue
        comparison_rows.append({
            "atlas_scan"         : f"Scan {scan_num}",
            "vessel_voxels"      : atlas_data[scan_num]["stats"]["vessel_voxels_after"],
            "mean_pct_10mm"      : round(ok["percentage_10mm"].mean(), 2),
            "median_pct_10mm"    : round(ok["percentage_10mm"].median(), 2),
            "passing_count"      : int((ok["percentage_10mm"] >= TARGET_PERCENTAGE).sum()),
            "total_scans"        : len(ok),
            "mean_time_s"        : round(ok["total_time_s"].mean(), 2),
            "mean_mi_metric"     : round(ok["final_metric"].abs().mean(), 4),
            "mean_dist_mm"       : round(ok["dist_mean_mm"].mean(), 2),
        })

    comparison_df  = pd.DataFrame(comparison_rows)
    comparison_csv = EXP2_OUT / "comparison_table.csv"
    comparison_df.to_csv(comparison_csv, index=False)

    # ── Step 5: Comparison figure ──────────────────────────────────────────
    save_comparison_figure(all_results, EXP2_OUT / "experiment2_report.png")

    # ── Final summary ──────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"EXPERIMENT 2 RESULTS")
    print(f"{'='*65}")
    print(comparison_df.to_string(index=False))

    best_row = comparison_df.loc[comparison_df["mean_pct_10mm"].idxmax()]
    print(f"\n  Best atlas : {best_row['atlas_scan']}")
    print(f"  Mean % 10mm: {best_row['mean_pct_10mm']:.1f}%")
    print(f"  Passing    : {best_row['passing_count']}/{best_row['total_scans']}")

    baseline = comparison_df[comparison_df["atlas_scan"] == "Scan 100"]
    if len(baseline) > 0:
        improvement = best_row["mean_pct_10mm"] - baseline["mean_pct_10mm"].values[0]
        print(f"  vs Scan 100: {improvement:+.1f}% improvement")

    print(f"\n✅ Results → {comparison_csv}")
    print(f"   Figure   → {EXP2_OUT}/experiment2_report.png")


if __name__ == "__main__":
    run_experiment2()