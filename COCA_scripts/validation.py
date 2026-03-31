"""
validation.py
─────────────
Step 3 of Part 2: Validate atlas registration quality.

Validation metric:
  % of calcium voxels within ±10mm of transformed vessel zones
  Target: >70%

Method:
  1. Load warped vessel mask (atlas_seg warped to COCA space)
  2. Compute Euclidean Distance Transform (EDT) on vessel mask
     → every voxel gets its distance in mm to nearest vessel voxel
  3. Load calcium mask for the same scan
  4. For each calcium voxel, look up its distance in the EDT
  5. Count how many calcium voxels have distance <= 10mm
  6. Divide by total calcium voxels → percentage


Outputs:
  - validation_output/
      ├── {scan_id}/
      │    ├── validation_meta.json     
      │    └── overlay_slice.png        
      ├── validation_results.csv        
      └── validation_report.png         

Usage:
    python validation.py
"""

import json
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy.ndimage import distance_transform_edt


SPLIT_CSV   = Path(r"...\COCA_output\data_canonical\tables\split_index.csv")
RESAMPLED   = Path(r"...\COCA_output\data_resampled")
REG_OUT     = Path(r"...\COCA_output\registration_output")
VAL_OUT     = Path(r"...\COCA_output\validation_output")
REG_CSV     = REG_OUT / "registration_results.csv"

# ] Validation Parameters 
DISTANCE_THRESHOLD_MM = 10.0   # ±10/15mm tolerance
TARGET_PERCENTAGE     = 70.0   # >70% target


def validate_scan(scan_id: str, voxel_spacing: list) -> dict:
    """
    Validates registration for a single scan.

    Args:
        scan_id       : scan identifier
        voxel_spacing : [X, Y, Z] spacing in mm — needed for EDT

    Returns:
        dict with validation metrics
    """
    #  Load warped vessel mask (atlas in COCA space) 
    warped_seg_path = REG_OUT / scan_id / "warped_atlas_seg.nii.gz"
    calcium_path    = RESAMPLED / scan_id / f"{scan_id}_seg.nii.gz"

    if not warped_seg_path.exists():
        return {"status": "missing_warped_seg"}
    if not calcium_path.exists():
        return {"status": "missing_calcium"}

    warped_seg  = sitk.GetArrayFromImage(
        sitk.ReadImage(str(warped_seg_path))
    ).astype(np.uint8)
    calcium_seg = sitk.GetArrayFromImage(
        sitk.ReadImage(str(calcium_path))
    ).astype(np.uint8)

    #  Align shapes if needed 
    # Warped seg should match calcium seg shape, crop/pad if minor mismatch
    if warped_seg.shape != calcium_seg.shape:
        min_shape = tuple(min(a, b) for a, b in
                          zip(warped_seg.shape, calcium_seg.shape))
        warped_seg  = warped_seg[:min_shape[0], :min_shape[1], :min_shape[2]]
        calcium_seg = calcium_seg[:min_shape[0], :min_shape[1], :min_shape[2]]

    #  Compute EDT on vessel mask 
    # scipy EDT: distance from background (0) to nearest foreground (1)
    # → invert: EDT on (1 - vessel_mask)
    vessel_binary = (warped_seg > 0).astype(np.uint8)
    vessel_voxels = int(vessel_binary.sum())

    if vessel_voxels == 0:
        return {"status": "no_vessel_voxels"}

    # Distance transform, sampling parameter converts voxels to mm
    # spacing order for scipy is (Z, Y, X) matching numpy array order
    spacing_zyx = [voxel_spacing[2], voxel_spacing[1], voxel_spacing[0]]
    dist_map_mm = distance_transform_edt(
        1 - vessel_binary,
        sampling=spacing_zyx
    )

    #  Compute validation metric 
    calcium_binary  = (calcium_seg > 0).astype(np.uint8)
    total_calcium   = int(calcium_binary.sum())

    if total_calcium == 0:
        return {"status": "no_calcium_voxels"}

    # Distance of each calcium voxel to nearest vessel voxel
    calcium_distances = dist_map_mm[calcium_binary == 1]

    # Count within threshold
    within_10mm = int((calcium_distances <= DISTANCE_THRESHOLD_MM).sum())
    percentage  = 100.0 * within_10mm / total_calcium

    # Additional distance statistics
    dist_mean   = float(calcium_distances.mean())
    dist_median = float(np.median(calcium_distances))
    dist_p25    = float(np.percentile(calcium_distances, 25))
    dist_p75    = float(np.percentile(calcium_distances, 75))
    dist_max    = float(calcium_distances.max())

    return {
        "status"          : "success",
        "total_calcium"   : total_calcium,
        "vessel_voxels"   : vessel_voxels,
        "within_10mm"     : within_10mm,
        "percentage_10mm" : round(percentage, 2),
        "passes_target"   : percentage >= TARGET_PERCENTAGE,
        "dist_mean_mm"    : round(dist_mean, 2),
        "dist_median_mm"  : round(dist_median, 2),
        "dist_p25_mm"     : round(dist_p25, 2),
        "dist_p75_mm"     : round(dist_p75, 2),
        "dist_max_mm"     : round(dist_max, 2),
        "warped_seg"      : warped_seg,
        "calcium_seg"     : calcium_seg,
        "dist_map_mm"     : dist_map_mm,
    }


#  Visual Overlay 

def save_overlay(scan_id, category, result, img_path, out_path):
    """
    Saves a 3-panel axial overlay figure:
      Panel 1: CT slice
      Panel 2: CT + calcium mask (red)
      Panel 3: CT + calcium (red) + vessel zone within 10mm (green)
    """
    # Load CT for background
    ct_img = sitk.GetArrayFromImage(
        sitk.ReadImage(str(img_path))
    ).astype(np.float32)

    warped_seg  = result["warped_seg"]
    calcium_seg = result["calcium_seg"]
    dist_map    = result["dist_map_mm"]

    # Align CT shape
    min_shape = tuple(min(ct_img.shape[i], calcium_seg.shape[i])
                      for i in range(3))
    ct_img     = ct_img[:min_shape[0], :min_shape[1], :min_shape[2]]
    calcium_seg = calcium_seg[:min_shape[0], :min_shape[1], :min_shape[2]]
    dist_map    = dist_map[:min_shape[0], :min_shape[1], :min_shape[2]]
    warped_seg  = warped_seg[:min_shape[0], :min_shape[1], :min_shape[2]]

    # Find best slice,  most calcium voxels
    calcium_per_slice = calcium_seg.sum(axis=(1, 2))
    if calcium_per_slice.max() == 0:
        best_slice = ct_img.shape[0] // 2
    else:
        best_slice = int(calcium_per_slice.argmax())

    ct_slice       = ct_img[best_slice]
    calcium_slice  = calcium_seg[best_slice]
    vessel_slice   = warped_seg[best_slice]
    dist_slice     = dist_map[best_slice]
    within_slice   = (dist_slice <= DISTANCE_THRESHOLD_MM).astype(np.uint8)

    # Normalize CT for display
    ct_norm = np.clip(ct_slice, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.patch.set_facecolor("white")

    titles = [
        "CT Slice (HU windowed)",
        "CT + Calcium Mask",
        f"CT + Calcium + Vessel Zone (±{DISTANCE_THRESHOLD_MM:.0f}mm)"
    ]

    for ax, title in zip(axes, titles):
        ax.imshow(ct_norm, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=9, pad=6)
        ax.axis("off")

    # Panel 2 calcium overlay
    if calcium_slice.max() > 0:
        ca_masked = np.ma.masked_where(calcium_slice == 0, calcium_slice)
        axes[1].imshow(ca_masked, cmap="autumn", alpha=0.8, vmin=0, vmax=1)

    # Panel 3: vessel zone+ calcium 
    if within_slice.max() > 0:
        zone_masked = np.ma.masked_where(within_slice == 0, within_slice)
        axes[2].imshow(zone_masked, cmap="Greens", alpha=0.35, vmin=0, vmax=1)
    if calcium_slice.max() > 0:
        ca_masked2 = np.ma.masked_where(calcium_slice == 0, calcium_slice)
        axes[2].imshow(ca_masked2, cmap="autumn", alpha=0.8, vmin=0, vmax=1)

    # Legend for panel 3
    legend_elements = [
        mpatches.Patch(facecolor="#22c55e", alpha=0.5,
                       label=f"Vessel zone (±{DISTANCE_THRESHOLD_MM:.0f}mm)"),
        mpatches.Patch(facecolor="#ef4444", alpha=0.8,
                       label="Calcium deposits"),
    ]
    axes[2].legend(handles=legend_elements, loc="lower right",
                   fontsize=7, framealpha=0.8)

    pct   = result["percentage_10mm"]
    color = "#16a34a" if result["passes_target"] else "#dc2626"
    fig.suptitle(
        f"Scan: {scan_id}  |  Category: {category}  |  "
        f"Slice: {best_slice}  |  "
        f"Calcium within ±10mm: {pct:.1f}%  "
        f"({'PASS' if result['passes_target'] else 'FAIL'})",
        fontsize=10, fontweight="bold", color=color
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(str(out_path), dpi=120, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)



def save_summary_figure(results_df, out_path):
    """
    4-panel summary figure:
      1. % calcium within 10mm per scan (bar chart)
      2. Distribution of calcium distances (histogram)
      3. Per-category pass rate
      4. Registration metric vs validation percentage (scatter)
    """
    df = results_df[results_df["status"] == "success"].copy()

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.patch.set_facecolor("white")

    CAT_COLORS = {
        "Zero": "#9ca3af", "Minimal": "#60a5fa",
        "Mild": "#34d399", "Moderate": "#f59e0b", "Severe": "#f87171"
    }

    # Panel 1: % within 10mm per scan 
    ax = axes[0, 0]
    colors = [
        "#16a34a" if p >= TARGET_PERCENTAGE else "#dc2626"
        for p in df["percentage_10mm"]
    ]
    bars = ax.bar(range(len(df)), df["percentage_10mm"],
                  color=colors, edgecolor="white", width=0.7)
    ax.axhline(TARGET_PERCENTAGE, color="#6366f1", linewidth=1.5,
               linestyle="--", label=f"Target ({TARGET_PERCENTAGE:.0f}%)")
    ax.set_xlabel("Scan index", fontsize=9)
    ax.set_ylabel("% calcium within ±10mm", fontsize=9)
    ax.set_title("Validation: % Calcium Within ±10mm of Vessel Zone",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 105)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    pass_count = int((df["percentage_10mm"] >= TARGET_PERCENTAGE).sum())
    ax.text(0.98, 0.05,
            f"Pass: {pass_count}/{len(df)}",
            transform=ax.transAxes, ha="right", fontsize=9,
            color="#16a34a" if pass_count == len(df) else "#dc2626",
            fontweight="bold")

    # Panel 2: Distance distribution histogram 
    ax = axes[0, 1]
    all_means   = df["dist_mean_mm"].values
    all_medians = df["dist_median_mm"].values
    ax.hist(all_means, bins=15, color="#60a5fa", alpha=0.7,
            edgecolor="white", label="Mean dist (mm)")
    ax.hist(all_medians, bins=15, color="#34d399", alpha=0.7,
            edgecolor="white", label="Median dist (mm)")
    ax.axvline(DISTANCE_THRESHOLD_MM, color="#dc2626", linewidth=1.5,
               linestyle="--", label=f"±{DISTANCE_THRESHOLD_MM:.0f}mm threshold")
    ax.set_xlabel("Distance to nearest vessel (mm)", fontsize=9)
    ax.set_ylabel("Number of scans", fontsize=9)
    ax.set_title("Calcium-to-Vessel Distance Distribution", fontsize=10,
                 fontweight="bold")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Panel 3: Per-category pass rate 
    ax = axes[1, 0]
    categories = ["Minimal", "Mild", "Moderate", "Severe"]
    pass_rates = []
    counts     = []
    for cat in categories:
        cat_df = df[df["category"] == cat]
        if len(cat_df) == 0:
            pass_rates.append(0)
            counts.append(0)
        else:
            rate = 100 * (cat_df["percentage_10mm"] >= TARGET_PERCENTAGE).mean()
            pass_rates.append(rate)
            counts.append(len(cat_df))

    bars = ax.bar(categories, pass_rates,
                  color=[CAT_COLORS[c] for c in categories],
                  edgecolor="white", width=0.5)
    for bar, count, rate in zip(bars, counts, pass_rates):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{rate:.0f}%\n(n={count})",
                ha="center", va="bottom", fontsize=8)
    ax.axhline(TARGET_PERCENTAGE, color="#6366f1", linewidth=1.5,
               linestyle="--", label=f"Target ({TARGET_PERCENTAGE:.0f}%)")
    ax.set_ylabel("Pass rate (%)", fontsize=9)
    ax.set_title("Pass Rate by CAC Category", fontsize=10,
                 fontweight="bold")
    ax.set_ylim(0, 115)
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    #  Panel 4: Registration metric vs validation % 
    ax = axes[1, 1]
    for cat in categories:
        cat_df = df[df["category"] == cat]
        if len(cat_df) > 0:
            ax.scatter(
                cat_df["final_metric"].abs(),
                cat_df["percentage_10mm"],
                c=CAT_COLORS[cat], label=cat,
                s=60, edgecolors="white", linewidths=0.5, zorder=3
            )
    ax.axhline(TARGET_PERCENTAGE, color="#dc2626", linewidth=1,
               linestyle="--", alpha=0.7)
    ax.set_xlabel("|MI metric| (higher = better alignment)", fontsize=9)
    ax.set_ylabel("% calcium within ±10mm", fontsize=9)
    ax.set_title("Registration Quality vs Validation Score", fontsize=10,
                 fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(linestyle="--", alpha=0.3)

    plt.tight_layout(pad=2.0)
    plt.savefig(str(out_path), dpi=130, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"  Summary figure → {out_path}")



def run_validation():
    print("=" * 60)
    print("  VALIDATION PIPELINE — Step 3 of Part 2")
    print("=" * 60)

    VAL_OUT.mkdir(parents=True, exist_ok=True)

    # Load successful registrations 
    reg_df     = pd.read_csv(REG_CSV)
    successful = reg_df[reg_df["status"] == "success"].reset_index(drop=True)
    print(f"\nSuccessful registrations: {len(successful)}")

    #  Load split index for category + spacing info 
    split_df = pd.read_csv(SPLIT_CSV)
    scan_info = split_df.set_index("scan_id")

    # Validation loop 
    val_results = []
    print(f"\n{'─'*70}")
    print(f"{'Scan':<15} {'Cat':<10} {'Ca vox':>8} {'Within10':>9} "
          f"{'%':>7} {'MeanDist':>9} {'Pass':>6}")
    print(f"{'─'*70}")

    for _, row in successful.iterrows():
        scan_id  = row["scan_id"]
        category = row["category"]

        # Get voxel spacing from resampled image
        img_path = RESAMPLED / scan_id / f"{scan_id}_img.nii.gz"
        img      = sitk.ReadImage(str(img_path))
        spacing  = list(img.GetSpacing())   # [X, Y, Z]

        result = validate_scan(scan_id, spacing)

        if result["status"] != "success":
            print(f"{scan_id:<15} {category:<10} "
                  f"{'SKIPPED: ' + result['status']:>30}")
            val_results.append({
                "scan_id" : scan_id,
                "category": category,
                "status"  : result["status"]
            })
            continue

        pct    = result["percentage_10mm"]
        passed = "Passed" if result["passes_target"] else "Fail"

        print(f"{scan_id:<15} {category:<10} "
              f"{result['total_calcium']:>8} "
              f"{result['within_10mm']:>9} "
              f"{pct:>6.1f}% "
              f"{result['dist_mean_mm']:>8.1f}mm "
              f"{passed:>6}")

        #  Save per-scan overlay 
        scan_val_out = VAL_OUT / scan_id
        scan_val_out.mkdir(parents=True, exist_ok=True)

        overlay_path = scan_val_out / "overlay_slice.png"
        save_overlay(scan_id, category, result, img_path, overlay_path)

        #  Save per-scan metadata 
        meta = {
            "scan_id"         : scan_id,
            "category"        : category,
            "total_calcium"   : result["total_calcium"],
            "vessel_voxels"   : result["vessel_voxels"],
            "within_10mm"     : result["within_10mm"],
            "percentage_10mm" : result["percentage_10mm"],
            "passes_target"   : result["passes_target"],
            "dist_mean_mm"    : result["dist_mean_mm"],
            "dist_median_mm"  : result["dist_median_mm"],
            "dist_p25_mm"     : result["dist_p25_mm"],
            "dist_p75_mm"     : result["dist_p75_mm"],
            "dist_max_mm"     : result["dist_max_mm"],
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

    #  Save results CSV 
    val_df      = pd.DataFrame(val_results)
    val_csv     = VAL_OUT / "validation_results.csv"
    val_df.to_csv(val_csv, index=False)

    #  Summary figures 
    print(f"\nGenerating summary figures...")
    success_df = val_df[val_df["status"] == "success"]
    save_summary_figure(success_df, VAL_OUT / "validation_report.png")

    #  Console summary 
    print(f"\n{'='*60}")
    print(f"VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Scans validated       : {len(success_df)}")
    passed = success_df[success_df["passes_target"] == True]
    print(f"  Scans passing >70%    : {len(passed)} / {len(success_df)}")
    print(f"  Mean % within 10mm    : {success_df['percentage_10mm'].mean():.1f}%")
    print(f"  Median % within 10mm  : {success_df['percentage_10mm'].median():.1f}%")
    print(f"  Min % within 10mm     : {success_df['percentage_10mm'].min():.1f}%")
    print(f"  Max % within 10mm     : {success_df['percentage_10mm'].max():.1f}%")
    print(f"\n  Per-category breakdown:")
    for cat in ["Minimal", "Mild", "Moderate", "Severe"]:
        cat_df = success_df[success_df["category"] == cat]
        if len(cat_df) > 0:
            cat_pass = (cat_df["percentage_10mm"] >= TARGET_PERCENTAGE).sum()
            print(f"    {cat:<12}: {cat_pass}/{len(cat_df)} pass  "
                  f"mean={cat_df['percentage_10mm'].mean():.1f}%")
    print(f"\nResults saved → {val_csv}")
    print(f"   Overlays    → {VAL_OUT}/{{scan_id}}/overlay_slice.png")
    print(f"   Report      → {VAL_OUT}/validation_report.png")


if __name__ == "__main__":
    run_validation()
