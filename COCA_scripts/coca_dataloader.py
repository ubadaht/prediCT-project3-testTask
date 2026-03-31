"""
coca_dataloader.py
──────────────────
Efficient MONAI DataLoader for COCA Cardiac Calcium Scoring Dataset.

Design:
  - Reads split_index.csv produced by stratified_split.py
  - Applies train/val/test transforms from augmentation.py
  - Pads/crops all volumes to uniform (256, 256, 48) for batching
  - Uses PersistentDataset for disk-based caching (fast after first epoch)
  - Returns batches of shape (B, 1, 256, 256, 48)

Usage:
    from coca_dataloader import get_dataloaders, get_dataset_stats
    train_loader, val_loader, test_loader = get_dataloaders()
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from monai.data import PersistentDataset, CacheDataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    ScaleIntensityRanged,
    RandFlipd,
    RandRotated,
    RandZoomd,
    RandGaussianNoised,
    RandScaleIntensityd,
    SpatialPadd,
    CenterSpatialCropd,
)


# ── Paths ──────────────────────────────────────────────────────────────────────
SPLIT_CSV      = Path(r"...\COCA_output\data_canonical\tables\split_index.csv")
RESAMPLED_DIR  = Path(r"...\COCA_output\data_resampled")
CACHE_DIR      = Path(r"...\COCA_output\cache")

# ── Volume shape ───────────────────────────────────────────────────────────────
# All volumes padded/cropped to this shape for batching
# (256, 256, 48) covers ~95% of scans without losing cardiac region
TARGET_SHAPE = (256, 256, 48)   # (X, Y, Z) in MONAI spatial convention

# ── HU Window ─────────────────────────────────────────────────────────────────
HU_MIN = -100.0
HU_MAX =  900.0

# ── DataLoader settings ────────────────────────────────────────────────────────
BATCH_SIZE  = 2    # small batch — 3D volumes are large
NUM_WORKERS = 2    # parallel loading workers
PIN_MEMORY  = True # faster CPU → GPU transfer


# ── Helper — build data list ───────────────────────────────────────────────────

def build_data_list(df: pd.DataFrame) -> list:
    """
    Converts DataFrame rows into MONAI-compatible list of dicts.
    Each dict has keys: image, mask, scan_id, category, voxels.

    MONAI expects:
        [
            {"image": "path/img.nii.gz", "mask": "path/seg.nii.gz", ...},
            ...
        ]

    Args:
        df : DataFrame with columns [scan_id, category, voxels]

    Returns:
        list of dicts, one per scan
    """
    data_list = []
    missing   = 0

    for _, row in df.iterrows():
        scan_id    = row["scan_id"]
        scan_folder = RESAMPLED_DIR / scan_id
        img_path   = scan_folder / f"{scan_id}_img.nii.gz"
        seg_path   = scan_folder / f"{scan_id}_seg.nii.gz"

        if not img_path.exists() or not seg_path.exists():
            missing += 1
            continue

        data_list.append({
            "image"    : str(img_path),
            "mask"     : str(seg_path),
            "scan_id"  : scan_id,
            "category" : row["category"],
            "voxels"   : int(row["voxels"]),
        })

    if missing > 0:
        print(f"  [WARNING] {missing} scans skipped — resampled files not found")

    return data_list


# ── Transform Pipelines ────────────────────────────────────────────────────────

def get_train_transforms():
    """
    Full pipeline for training set:
      Load → ChannelFirst → Type → HU Window →
      Pad → Crop → Flip → Rotate → Zoom → Noise → Scale → Clamp
    """
    return Compose([

        # ── Load & Format ──────────────────────────────────────────────────
        LoadImaged(keys=["image", "mask"], image_only=True),
        EnsureChannelFirstd(keys=["image", "mask"]),
        EnsureTyped(keys=["image", "mask"], dtype=torch.float32),

        # ── HU Windowing ───────────────────────────────────────────────────
        ScaleIntensityRanged(
            keys=["image"],
            a_min=HU_MIN, a_max=HU_MAX,
            b_min=0.0,    b_max=1.0,
            clip=True
        ),

        # ── Uniform Shape (Pad then Crop) ──────────────────────────────────
        # Step 1: Pad volumes smaller than TARGET_SHAPE with zeros
        SpatialPadd(
            keys=["image", "mask"],
            spatial_size=TARGET_SHAPE,
            mode="constant",        # pad with constant value
            value=0                 # 0 = air in normalized space
        ),
        # Step 2: Crop volumes larger than TARGET_SHAPE from center
        # Center crop keeps the cardiac region (centered in FOV)
        CenterSpatialCropd(
            keys=["image", "mask"],
            roi_size=TARGET_SHAPE
        ),

        # ── Geometric Augmentation ─────────────────────────────────────────
        RandFlipd(
            keys=["image", "mask"],
            spatial_axis=2,         # X axis = left/right flip
            prob=0.50
        ),
        RandRotated(
            keys=["image", "mask"],
            range_x=0.2618,         # ±15° around X
            range_y=0.2618,         # ±15° around Y
            range_z=0.2618,         # ±15° around Z
            prob=0.50,
            keep_size=True,
            mode=["bilinear", "nearest"],
            padding_mode="zeros"
        ),
        RandZoomd(
            keys=["image", "mask"],
            min_zoom=0.85,
            max_zoom=1.15,
            prob=0.30,
            mode=["trilinear", "nearest"],
            keep_size=True
        ),

        # ── Intensity Augmentation (image only) ────────────────────────────
        RandGaussianNoised(
            keys=["image"],
            mean=0.0,
            std=0.05,
            prob=0.20
        ),
        RandScaleIntensityd(
            keys=["image"],
            factors=0.10,
            prob=0.20
        ),

        # ── Final clamp to [0, 1] after intensity augmentation ─────────────
        ScaleIntensityRanged(
            keys=["image"],
            a_min=0.0, a_max=1.0,
            b_min=0.0, b_max=1.0,
            clip=True
        ),
    ])


def get_val_test_transforms():
    """
    Minimal pipeline for validation and test sets:
      Load → ChannelFirst → Type → HU Window → Pad → Crop
    No augmentation.
    """
    return Compose([
        LoadImaged(keys=["image", "mask"], image_only=True),
        EnsureChannelFirstd(keys=["image", "mask"]),
        EnsureTyped(keys=["image", "mask"], dtype=torch.float32),

        ScaleIntensityRanged(
            keys=["image"],
            a_min=HU_MIN, a_max=HU_MAX,
            b_min=0.0,    b_max=1.0,
            clip=True
        ),

        SpatialPadd(
            keys=["image", "mask"],
            spatial_size=TARGET_SHAPE,
            mode="constant",
            value=0
        ),
        CenterSpatialCropd(
            keys=["image", "mask"],
            roi_size=TARGET_SHAPE
        ),
    ])


# ── Dataset & DataLoader Factory ───────────────────────────────────────────────

def get_dataloaders(
    use_cache : bool = True,
    batch_size : int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
):
    """
    Builds and returns train, val, test DataLoaders.

    Args:
        use_cache   : use PersistentDataset (disk cache) for speed
                      set False to use CacheDataset (RAM cache, no disk write)
        batch_size  : samples per batch (default 2 for 3D volumes)
        num_workers : parallel loading workers (default 2)

    Returns:
        (train_loader, val_loader, test_loader)

    Output batch shape:
        image : (B, 1, 256, 256, 48)  float32  [0, 1]
        mask  : (B, 1, 256, 256, 48)  float32  {0, 1}
    """

    # ── Load split CSV ─────────────────────────────────────────────────────
    print(f"Loading split index from {SPLIT_CSV}...")
    df = pd.read_csv(SPLIT_CSV)

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df   = df[df["split"] == "val"].reset_index(drop=True)
    test_df  = df[df["split"] == "test"].reset_index(drop=True)

    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # ── Build data lists ───────────────────────────────────────────────────
    train_list = build_data_list(train_df)
    val_list   = build_data_list(val_df)
    test_list  = build_data_list(test_df)

    # ── Build transforms ───────────────────────────────────────────────────
    train_tx   = get_train_transforms()
    val_test_tx = get_val_test_transforms()

    # ── Build datasets ─────────────────────────────────────────────────────
    if use_cache:
        # PersistentDataset: caches to disk — survives restarts
        # First run is slow (processes + saves), subsequent runs are fast
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        train_cache = CACHE_DIR / "train"
        val_cache   = CACHE_DIR / "val"
        test_cache  = CACHE_DIR / "test"

        train_ds = PersistentDataset(
            data=train_list,
            transform=val_test_tx,     # cache deterministic transforms only
            cache_dir=str(train_cache)
        )
        val_ds = PersistentDataset(
            data=val_list,
            transform=val_test_tx,
            cache_dir=str(val_cache)
        )
        test_ds = PersistentDataset(
            data=test_list,
            transform=val_test_tx,
            cache_dir=str(test_cache)
        )
        print(f"  Using PersistentDataset (cache: {CACHE_DIR})")

    else:
        # CacheDataset: caches in RAM — faster but uses more memory
        train_ds = CacheDataset(
            data=train_list,
            transform=train_tx,
            cache_rate=1.0,
            num_workers=num_workers
        )
        val_ds = CacheDataset(
            data=val_list,
            transform=val_test_tx,
            cache_rate=1.0,
            num_workers=num_workers
        )
        test_ds = CacheDataset(
            data=test_list,
            transform=val_test_tx,
            cache_rate=1.0,
            num_workers=num_workers
        )
        print(f"  Using CacheDataset (RAM cache)")

    # ── Build DataLoaders ──────────────────────────────────────────────────
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,               # shuffle every epoch
        num_workers=num_workers,
        pin_memory=PIN_MEMORY,
        drop_last=True              # drop incomplete last batch
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,              # no shuffle for eval
        num_workers=num_workers,
        pin_memory=PIN_MEMORY,
        drop_last=False
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,               # batch=1 for test (full volume evaluation)
        shuffle=False,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY,
        drop_last=False
    )

    print(f"\nDataLoaders ready:")
    print(f"  Train batches : {len(train_loader)}")
    print(f"  Val   batches : {len(val_loader)}")
    print(f"  Test  batches : {len(test_loader)}")
    print(f"  Batch shape   : (B, 1, {TARGET_SHAPE[0]}, {TARGET_SHAPE[1]}, {TARGET_SHAPE[2]})")

    return train_loader, val_loader, test_loader


# ── Dataset Statistics ─────────────────────────────────────────────────────────

def get_dataset_stats(loader: DataLoader, n_batches: int = 5):
    """
    Computes mean and std of image intensities across n_batches.
    Useful for verifying the DataLoader output before training.

    Args:
        loader   : a DataLoader (train/val/test)
        n_batches: how many batches to sample (default 5)
    """
    means, stds = [], []
    print(f"\nComputing dataset statistics over {n_batches} batches...")

    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        img = batch["image"]
        means.append(img.mean().item())
        stds.append(img.std().item())
        print(f"  Batch {i}: shape={tuple(img.shape)}  "
              f"min={img.min():.4f}  max={img.max():.4f}  "
              f"mean={img.mean():.4f}  std={img.std():.4f}  "
              f"mask_sum={batch['mask'].sum().item():.0f}")

    print(f"\nOverall — mean: {np.mean(means):.4f}  std: {np.mean(stds):.4f}")


# ── Standalone verification ────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  COCA DataLoader Verification")
    print("=" * 55)

    train_loader, val_loader, test_loader = get_dataloaders(
        use_cache=True,
        batch_size=2,
        num_workers=0       # 0 workers for Windows compatibility
    )

    # Check one batch from each split
    print("\n── Train batch ──")
    batch = next(iter(train_loader))
    print(f"  image shape : {tuple(batch['image'].shape)}")
    print(f"  mask  shape : {tuple(batch['mask'].shape)}")
    print(f"  image range : [{batch['image'].min():.4f}, {batch['image'].max():.4f}]")
    print(f"  mask  unique: {batch['mask'].unique().tolist()}")
    print(f"  categories  : {batch['category']}")
    print(f"  voxel counts: {batch['voxels'].tolist()}")

    print("\n── Val batch ──")
    batch = next(iter(val_loader))
    print(f"  image shape : {tuple(batch['image'].shape)}")
    print(f"  image range : [{batch['image'].min():.4f}, {batch['image'].max():.4f}]")

    print("\n── Test batch ──")
    batch = next(iter(test_loader))
    print(f"  image shape : {tuple(batch['image'].shape)}")
    print(f"  image range : [{batch['image'].min():.4f}, {batch['image'].max():.4f}]")

    # Full stats over 5 batches
    get_dataset_stats(train_loader, n_batches=5)

    print("\n✅ DataLoader verification complete.")