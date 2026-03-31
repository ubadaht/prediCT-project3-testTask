"""
Stratified Train/Val/Test Split for COCA Calcium Scoring Dataset.

Workflow:
  - Map raw voxel counts → 5 CAC clinical categories
  - Split 70/15/15 within each category (preserves class proportions)
  - Flag 20-30 test scans as Part 2 (registration) candidates
  - Save split_index.csv for use by DataLoader

Usage:
    python stratified_split.py
    from stratified_split import load_splits
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter


CSV_IN  = Path(r"...\COCA_output\data_canonical\tables\scan_index.csv")
CSV_OUT = Path(r"...\COCA_output\data_canonical\tables\split_index.csv")

#  Split ratios 
VAL_RATIO  = 0.15   # 15% validation
TEST_RATIO = 0.15   # 15% test
# Train = remaining 70%

#  CAC Category boundaries (voxel counts) 
CAC_BINS   = [0, 1, 101, 501, 1501, np.inf]
CAC_LABELS = ["Zero", "Minimal", "Mild", "Moderate", "Severe"]

#  Part 2 registration candidates 
N_PART2 = 25   # how many scans to flag for atlas registration



def assign_category(voxels: pd.Series) -> pd.Series:
    """
    Maps raw calcium voxel counts to clinical CAC categories.

    Bins:
      Zero     : voxels = 0
      Minimal  : voxels 1   – 100
      Mild     : voxels 101 – 500
      Moderate : voxels 501 – 1500
      Severe   : voxels > 1500

    Args:
        voxels : pandas Series of integer voxel counts

    Returns:
        pandas Series of category strings
    """
    return pd.cut(
        voxels,
        bins=CAC_BINS,
        labels=CAC_LABELS,
        right=False       # intervals are [left, right)
    ).astype(str)


def stratified_split(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs two-stage stratified split:
      Stage 1: Full dataset → (train+val) 85%  +  test 15%
      Stage 2: train+val    → train 82.4%      +  val  17.6%
               (82.4% of 85% ≈ 70% of total, 17.6% of 85% ≈ 15% of total)

    Stratification key: CAC category — ensures each split has
    proportional representation of all 5 calcium categories.

    Args:
        df : DataFrame with columns [scan_id, voxels, category]

    Returns:
        df with new column 'split' ∈ {train, val, test}
    """
    df = df.copy()
    df["split"] = ""

    #  Stage 1: Split off test set 
    sss1 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=TEST_RATIO,
        random_state=42        # fixed seed for reproducibility
    )
    trainval_idx, test_idx = next(sss1.split(df, df["category"]))

    df.iloc[test_idx, df.columns.get_loc("split")] = "test"

    #  Stage 2: Split train+val into train and val 
    df_trainval = df.iloc[trainval_idx].copy()

    # val ratio relative to trainval subset
    val_ratio_adjusted = VAL_RATIO / (1 - TEST_RATIO)

    sss2 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_ratio_adjusted,
        random_state=42
    )
    train_idx_rel, val_idx_rel = next(
        sss2.split(df_trainval, df_trainval["category"])
    )

    train_global = df_trainval.iloc[train_idx_rel].index
    val_global   = df_trainval.iloc[val_idx_rel].index

    df.loc[train_global, "split"] = "train"
    df.loc[val_global,   "split"] = "val"

    return df


def flag_part2_candidates(df: pd.DataFrame, n: int = N_PART2) -> pd.DataFrame:
    """
    Selects n scans from the TEST set as Part 2 registration candidates.

    Criteria:
      - Only from test split
      - Only non-zero calcium (registration validation needs calcium)
      - Balanced across non-zero categories (Minimal/Mild/Moderate/Severe)
      - Deterministic (random_state=42)

    Args:
        df : DataFrame with 'split' and 'category' columns
        n  : total number of Part 2 candidates (default 25)

    Returns:
        df with new boolean column 'part2_candidate'
    """
    df = df.copy()
    df["part2_candidate"] = False

    # Only consider test scans with calcium
    test_nonzero = df[
        (df["split"] == "test") &
        (df["category"] != "Zero")
    ].copy()

    non_zero_cats = ["Minimal", "Mild", "Moderate", "Severe"]
    per_category  = n // len(non_zero_cats)   # ~6 per category
    remainder     = n % len(non_zero_cats)    # distribute remainder to severe

    selected_indices = []
    for i, cat in enumerate(non_zero_cats):
        cat_scans = test_nonzero[test_nonzero["category"] == cat]
        # give remainder slots to the most severe
        n_select  = per_category + (1 if i >= len(non_zero_cats) - remainder else 0)
        n_select  = min(n_select, len(cat_scans))   # can't select more than available

        if n_select > 0:
            sampled = cat_scans.sample(n=n_select, random_state=42)
            selected_indices.extend(sampled.index.tolist())

    df.loc[selected_indices, "part2_candidate"] = True
    return df


def print_statistics(df: pd.DataFrame):
    """Prints split statistics to verify stratification worked correctly."""

    print("\n" + "═" * 60)
    print("  DATASET SPLIT STATISTICS")
    print("═" * 60)

    #  Overall split counts 
    split_counts = df["split"].value_counts()
    total        = len(df)
    print(f"\nOverall split (total = {total}):")
    for split in ["train", "val", "test"]:
        n   = split_counts.get(split, 0)
        pct = 100 * n / total
        print(f"  {split:<6}: {n:>4} scans  ({pct:.1f}%)")

    #  Category distribution per split 
    print(f"\nCategory distribution per split:")
    print(f"  {'Category':<12}", end="")
    for split in ["train", "val", "test", "TOTAL"]:
        print(f" {split:>8}", end="")
    print()
    print("  " + "─" * 50)

    for cat in CAC_LABELS:
        print(f"  {cat:<12}", end="")
        cat_total = 0
        for split in ["train", "val", "test"]:
            n = len(df[(df["split"] == split) & (df["category"] == cat)])
            cat_total += n
            print(f" {n:>8}", end="")
        print(f" {cat_total:>8}")

    #  Proportions check 
    print(f"\nCategory proportions per split (should be ~equal):")
    print(f"  {'Category':<12}", end="")
    for split in ["train", "val", "test"]:
        print(f" {split:>8}", end="")
    print()
    print("  " + "─" * 38)

    split_totals = {s: len(df[df["split"] == s]) for s in ["train", "val", "test"]}
    for cat in CAC_LABELS:
        print(f"  {cat:<12}", end="")
        for split in ["train", "val", "test"]:
            n   = len(df[(df["split"] == split) & (df["category"] == cat)])
            pct = 100 * n / split_totals[split] if split_totals[split] > 0 else 0
            print(f" {pct:>7.1f}%", end="")
        print()

    #  Part 2 candidates 
    part2 = df[df["part2_candidate"] == True]
    print(f"\nPart 2 (Registration) candidates: {len(part2)} scans")
    print(f"  Category breakdown:")
    for cat, count in part2["category"].value_counts().items():
        print(f"    {cat:<12}: {count} scans")

    #  Calcium voxel stats per split 
    print(f"\nCalcium voxel statistics per split:")
    print(f"  {'Split':<8} {'Mean':>8} {'Median':>8} {'Max':>8} {'Zero%':>8}")
    print("  " + "─" * 44)
    for split in ["train", "val", "test"]:
        sub      = df[df["split"] == split]["voxels"]
        zero_pct = 100 * (sub == 0).sum() / len(sub)
        print(f"  {split:<8} {sub.mean():>8.1f} {sub.median():>8.1f} "
              f"{sub.max():>8.0f} {zero_pct:>7.1f}%")

    print("\n" + "═" * 60)


def load_splits(csv_path: str = str(CSV_OUT)):
    """
    Utility function for DataLoader, loads split_index.csv and
    returns three DataFrames: train, val, test.

    Usage:
        from stratified_split import load_splits
        train_df, val_df, test_df = load_splits()

    Returns:
        (train_df, val_df, test_df) — each with columns:
        [patient_id, scan_id, voxels, category, split, part2_candidate, folder_path]
    """
    df    = pd.read_csv(csv_path)
    train = df[df["split"] == "train"].reset_index(drop=True)
    val   = df[df["split"] == "val"].reset_index(drop=True)
    test  = df[df["split"] == "test"].reset_index(drop=True)
    return train, val, test



if __name__ == "__main__":

    #  Load scan index 
    print(f"Loading {CSV_IN}...")
    df = pd.read_csv(CSV_IN)
    print(f"Loaded {len(df)} scans.")

    #  Assign CAC categories 
    df["category"] = assign_category(df["voxels"])
    print(f"\nCAC category distribution:")
    for cat, count in df["category"].value_counts().items():
        pct = 100 * count / len(df)
        print(f"  {cat:<12}: {count:>4} ({pct:.1f}%)")

    #  Perform stratified split 
    print(f"\nPerforming stratified 70/15/15 split...")
    df = stratified_split(df)

    #  Flag Part 2 candidates 
    print(f"Selecting {N_PART2} Part 2 registration candidates...")
    df = flag_part2_candidates(df, n=N_PART2)

    print_statistics(df)

    # Save 
    df.to_csv(CSV_OUT, index=False)
    print(f"\n Saved split_index.csv → {CSV_OUT}")
    print(f"   Columns: {df.columns.tolist()}")
