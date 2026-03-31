"""
dataset_statistics.py
─────────────────────
Computes and exports dataset statistics for the COCA preprocessing pipeline.
Produces:
  1. Console summary (tables)
  2. dataset_statistics.html — visual report with charts and overlays

Usage:
    python dataset_statistics.py
"""

import json
import numpy as np
import pandas as pd
import SimpleITK as sitk
from pathlib import Path
from collections import defaultdict
from pdf_report import save_pdf_report

# ── Paths ──────────────────────────────────────────────────────────────────────
SPLIT_CSV     = Path(r"...\COCA_output\data_canonical\tables\split_index.csv")
RESAMPLED_DIR = Path(r"...\COCA_output\data_resampled")
CANONICAL_DIR = Path(r"...\COCA_output\data_canonical\images")
OUT_HTML      = Path(r"...\COCA_output\dataset_statistics.html")

CAC_LABELS = ["Zero", "Minimal", "Mild", "Moderate", "Severe"]
COLORS     = {
    "Zero":     "#94a3b8",
    "Minimal":  "#60a5fa",
    "Mild":     "#34d399",
    "Moderate": "#f59e0b",
    "Severe":   "#f87171",
}
SPLIT_COLORS = {"train": "#6366f1", "val": "#22d3ee", "test": "#f472b6"}


# ── Data Collection ────────────────────────────────────────────────────────────

def collect_spacing_stats(df: pd.DataFrame, n_samples: int = 80):
    """Sample original spacings from canonical (pre-resample) NIfTI metadata."""
    spacings = []
    samples  = df.sample(min(n_samples, len(df)), random_state=42)

    for _, row in samples.iterrows():
        meta_candidates = list((CANONICAL_DIR / row["scan_id"]).glob("*_meta.json"))
        if meta_candidates:
            try:
                meta = json.loads(meta_candidates[0].read_text())
                sp   = meta.get("spacing", None)   # [x, y, z] mm
                if sp:
                    spacings.append(sp)
            except Exception:
                pass

    return spacings


def collect_volume_shapes(df: pd.DataFrame, n_samples: int = 80):
    """Sample resampled volume shapes."""
    shapes  = []
    samples = df.sample(min(n_samples, len(df)), random_state=42)

    for _, row in samples.iterrows():
        scan_id  = row["scan_id"]
        img_path = RESAMPLED_DIR / scan_id / f"{scan_id}_img.nii.gz"
        if img_path.exists():
            try:
                img = sitk.ReadImage(str(img_path))
                shapes.append(list(img.GetSize()))   # [X, Y, Z]
            except Exception:
                pass

    return shapes


def collect_intensity_stats(df: pd.DataFrame, n_samples: int = 30):
    """Sample per-scan intensity stats from resampled volumes."""
    stats   = []
    samples = df.sample(min(n_samples, len(df)), random_state=42)
    HU_MIN, HU_MAX = -100.0, 900.0

    for _, row in samples.iterrows():
        scan_id  = row["scan_id"]
        img_path = RESAMPLED_DIR / scan_id / f"{scan_id}_img.nii.gz"
        if img_path.exists():
            try:
                arr = sitk.GetArrayFromImage(sitk.ReadImage(str(img_path))).astype(np.float32)
                arr = np.clip(arr, HU_MIN, HU_MAX)
                arr = (arr - HU_MIN) / (HU_MAX - HU_MIN)
                stats.append({
                    "scan_id" : scan_id,
                    "mean"    : float(arr.mean()),
                    "std"     : float(arr.std()),
                    "min"     : float(arr.min()),
                    "max"     : float(arr.max()),
                    "category": row["category"],
                })
            except Exception:
                pass

    return stats


def get_example_slice(df: pd.DataFrame):
    """Get a non-zero calcium scan for the overlay example."""
    nonzero = df[df["voxels"] > 100].sample(1, random_state=7).iloc[0]
    scan_id  = nonzero["scan_id"]
    img_path = RESAMPLED_DIR / scan_id / f"{scan_id}_img.nii.gz"
    seg_path = RESAMPLED_DIR / scan_id / f"{scan_id}_seg.nii.gz"

    if not img_path.exists():
        return None, None, None, None

    img_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(img_path)))   # (Z,Y,X)
    seg_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(seg_path)))

    # Find slice with most calcium
    calcium_per_slice = seg_arr.sum(axis=(1, 2))
    best_z            = int(np.argmax(calcium_per_slice))

    img_slice = img_arr[best_z]
    seg_slice = seg_arr[best_z]

    # Normalize slice to 0-255 for display
    HU_MIN, HU_MAX = -100.0, 900.0
    img_norm = np.clip(img_slice.astype(np.float32), HU_MIN, HU_MAX)
    img_norm = ((img_norm - HU_MIN) / (HU_MAX - HU_MIN) * 255).astype(np.uint8)

    return img_norm, seg_slice, scan_id, nonzero["category"]


# ── Console Summary ────────────────────────────────────────────────────────────

def print_console_summary(df: pd.DataFrame):
    print("\n" + "═" * 62)
    print("  COCA DATASET STATISTICS SUMMARY")
    print("═" * 62)

    print(f"\n{'Total scans':<35}: {len(df)}")
    print(f"{'Scans with calcium (voxels > 0)':<35}: {(df['voxels'] > 0).sum()}")
    print(f"{'Zero calcium scans':<35}: {(df['voxels'] == 0).sum()}")
    print(f"{'Part 2 registration candidates':<35}: {df['part2_candidate'].sum()}")

    print(f"\nCalcium voxel statistics:")
    print(f"  {'Mean':<12}: {df['voxels'].mean():.1f}")
    print(f"  {'Std':<12}: {df['voxels'].std():.1f}")
    print(f"  {'Median':<12}: {df['voxels'].median():.1f}")
    print(f"  {'Min':<12}: {df['voxels'].min():.0f}")
    print(f"  {'Max':<12}: {df['voxels'].max():.0f}")

    print(f"\nPer-split breakdown:")
    print(f"  {'Split':<10} {'N':>6} {'%':>7} {'Mean vox':>10} {'Zero%':>8}")
    print("  " + "─" * 45)
    for split in ["train", "val", "test"]:
        sub      = df[df["split"] == split]
        pct      = 100 * len(sub) / len(df)
        zero_pct = 100 * (sub["voxels"] == 0).sum() / len(sub)
        print(f"  {split:<10} {len(sub):>6} {pct:>6.1f}% {sub['voxels'].mean():>10.1f} {zero_pct:>7.1f}%")

    print(f"\nCategory distribution:")
    print(f"  {'Category':<12} {'Total':>7} {'Train':>7} {'Val':>7} {'Test':>7} {'%Total':>8}")
    print("  " + "─" * 52)
    for cat in CAC_LABELS:
        total = len(df[df["category"] == cat])
        tr    = len(df[(df["split"] == "train") & (df["category"] == cat)])
        va    = len(df[(df["split"] == "val")   & (df["category"] == cat)])
        te    = len(df[(df["split"] == "test")  & (df["category"] == cat)])
        pct   = 100 * total / len(df)
        print(f"  {cat:<12} {total:>7} {tr:>7} {va:>7} {te:>7} {pct:>7.1f}%")

    print("\n" + "═" * 62)


# ── HTML Report ────────────────────────────────────────────────────────────────

def build_html_report(df: pd.DataFrame, spacings, shapes, intensity_stats, img_slice, seg_slice, example_id, example_cat):

    # ── Precompute data for charts ─────────────────────────────────────────
    # Category counts
    cat_counts  = {cat: int((df["category"] == cat).sum()) for cat in CAC_LABELS}
    cat_colors  = [COLORS[c] for c in CAC_LABELS]

    # Split counts
    split_counts = {s: int((df["split"] == s).sum()) for s in ["train", "val", "test"]}

    # Voxel histogram buckets (log scale bins)
    nonzero_vox = df[df["voxels"] > 0]["voxels"].values
    hist_bins   = [1, 10, 50, 100, 250, 500, 1000, 2000, 5000, 12000]
    hist_counts = []
    hist_labels = []
    for i in range(len(hist_bins) - 1):
        lo = hist_bins[i]; hi = hist_bins[i+1]
        cnt = int(((nonzero_vox >= lo) & (nonzero_vox < hi)).sum())
        hist_counts.append(cnt)
        hist_labels.append(f"{lo}–{hi}")

    # Per-split category proportions
    split_cat_data = {}
    for split in ["train", "val", "test"]:
        sub = df[df["split"] == split]
        split_cat_data[split] = [
            round(100 * len(sub[sub["category"] == cat]) / len(sub), 1)
            for cat in CAC_LABELS
        ]

    # Spacing stats
    if spacings:
        sp_arr = np.array(spacings)
        sp_xy_mean = float(np.mean(sp_arr[:, 0]))
        sp_z_mean  = float(np.mean(sp_arr[:, 2]))
        sp_xy_std  = float(np.std(sp_arr[:, 0]))
        sp_z_std   = float(np.std(sp_arr[:, 2]))
    else:
        sp_xy_mean = sp_z_mean = sp_xy_std = sp_z_std = 0.0

    # Shape stats
    if shapes:
        sh_arr   = np.array(shapes)
        sh_x_med = float(np.median(sh_arr[:, 0]))
        sh_z_med = float(np.median(sh_arr[:, 2]))
    else:
        sh_x_med = sh_z_med = 0.0

    # Intensity stats
    if intensity_stats:
        i_means = [s["mean"] for s in intensity_stats]
        i_mean  = float(np.mean(i_means))
        i_std   = float(np.std(i_means))
    else:
        i_mean = i_std = 0.0

    # Example slice → base64 PNG
    import base64, io
    slice_b64 = ""
    overlay_b64 = ""
    if img_slice is not None:
        try:
            from PIL import Image as PILImage
            # Grayscale CT slice
            pil_img  = PILImage.fromarray(img_slice, mode='L').convert('RGB')
            buf = io.BytesIO()
            pil_img.save(buf, format='PNG')
            slice_b64 = base64.b64encode(buf.getvalue()).decode()

            # Overlay: CT + red calcium mask
            rgb = np.stack([img_slice, img_slice, img_slice], axis=-1)
            if seg_slice is not None and seg_slice.max() > 0:
                mask_bool = seg_slice.astype(bool)
                rgb[mask_bool, 0] = 255
                rgb[mask_bool, 1] = 50
                rgb[mask_bool, 2] = 50
            pil_ov = PILImage.fromarray(rgb.astype(np.uint8), mode='RGB')
            buf2   = io.BytesIO()
            pil_ov.save(buf2, format='PNG')
            overlay_b64 = base64.b64encode(buf2.getvalue()).decode()
        except ImportError:
            pass

    # ── Build HTML ─────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>COCA Dataset Statistics</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg:      #0a0e1a;
    --surface: #111827;
    --border:  #1e2d40;
    --text:    #e2e8f0;
    --muted:   #64748b;
    --accent:  #38bdf8;
    --green:   #34d399;
    --orange:  #f59e0b;
    --red:     #f87171;
    --purple:  #a78bfa;
  }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'Syne', sans-serif;
    min-height: 100vh;
    padding: 2rem;
  }}
  .header {{
    border-bottom: 1px solid var(--border);
    padding-bottom: 1.5rem;
    margin-bottom: 2.5rem;
  }}
  .header h1 {{
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    color: var(--accent);
  }}
  .header p {{
    color: var(--muted);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    margin-top: 0.4rem;
  }}
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 1.5rem; }}
  .grid-3 {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1.5rem; margin-bottom: 1.5rem; }}
  .grid-4 {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 1.5rem; }}
  .card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
  }}
  .card h2 {{
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 1rem;
    font-family: 'JetBrains Mono', monospace;
  }}
  .stat-big {{
    font-size: 2.8rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    line-height: 1;
    color: var(--accent);
  }}
  .stat-label {{
    font-size: 0.75rem;
    color: var(--muted);
    margin-top: 0.3rem;
    font-family: 'JetBrains Mono', monospace;
  }}
  .kv-table {{ width: 100%; border-collapse: collapse; }}
  .kv-table td {{
    padding: 0.45rem 0;
    border-bottom: 1px solid var(--border);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
  }}
  .kv-table td:first-child {{ color: var(--muted); width: 55%; }}
  .kv-table td:last-child  {{ color: var(--text); text-align: right; font-weight: 700; }}
  .kv-table tr:last-child td {{ border-bottom: none; }}
  .chart-wrap {{ position: relative; height: 220px; }}
  .chart-wrap-tall {{ position: relative; height: 280px; }}
  .cat-badge {{
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    font-size: 0.7rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.05em;
  }}
  .split-row {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.5rem 0;
    border-bottom: 1px solid var(--border);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
  }}
  .split-row:last-child {{ border-bottom: none; }}
  .split-bar-wrap {{ width: 45%; background: var(--border); border-radius: 3px; height: 6px; }}
  .split-bar {{ height: 6px; border-radius: 3px; }}
  .img-pair {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }}
  .img-wrap {{ text-align: center; }}
  .img-wrap img {{ width: 100%; image-rendering: pixelated; border-radius: 6px; border: 1px solid var(--border); }}
  .img-wrap p {{ font-size: 0.7rem; color: var(--muted); margin-top: 0.4rem; font-family: 'JetBrains Mono', monospace; }}
  .section-title {{
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 1rem;
    margin-top: 0.5rem;
    font-family: 'JetBrains Mono', monospace;
    border-left: 3px solid var(--accent);
    padding-left: 0.7rem;
  }}
  canvas {{ max-width: 100%; }}
</style>
</head>
<body>

<div class="header">
  <h1>COCA Dataset Statistics</h1>
  <p>Coronary Calcium Scoring CT · Preprocessing Pipeline Report · {len(df)} scans</p>
</div>

<!-- KPI Row -->
<div class="grid-4">
  <div class="card">
    <h2>Total Scans</h2>
    <div class="stat-big">{len(df)}</div>
    <div class="stat-label">787 processed</div>
  </div>
  <div class="card">
    <h2>With Calcium</h2>
    <div class="stat-big" style="color:var(--green)">{int((df['voxels']>0).sum())}</div>
    <div class="stat-label">{100*(df['voxels']>0).mean():.1f}% of dataset</div>
  </div>
  <div class="card">
    <h2>Zero Calcium</h2>
    <div class="stat-big" style="color:var(--muted)">{int((df['voxels']==0).sum())}</div>
    <div class="stat-label">{100*(df['voxels']==0).mean():.1f}% of dataset</div>
  </div>
  <div class="card">
    <h2>Part 2 Candidates</h2>
    <div class="stat-big" style="color:var(--orange)">{int(df['part2_candidate'].sum())}</div>
    <div class="stat-label">for atlas registration</div>
  </div>
</div>

<!-- Charts Row 1 -->
<div class="grid-2">
  <div class="card">
    <h2>CAC Category Distribution</h2>
    <div class="chart-wrap">
      <canvas id="catChart"></canvas>
    </div>
  </div>
  <div class="card">
    <h2>Train / Val / Test Split</h2>
    <div class="chart-wrap">
      <canvas id="splitChart"></canvas>
    </div>
  </div>
</div>

<!-- Charts Row 2 -->
<div class="grid-2">
  <div class="card">
    <h2>Calcium Voxel Histogram (non-zero scans)</h2>
    <div class="chart-wrap-tall">
      <canvas id="histChart"></canvas>
    </div>
  </div>
  <div class="card">
    <h2>Category Proportions per Split</h2>
    <div class="chart-wrap-tall">
      <canvas id="stackChart"></canvas>
    </div>
  </div>
</div>

<!-- Stats Tables -->
<div class="grid-3">
  <div class="card">
    <h2>Calcium Voxel Statistics</h2>
    <table class="kv-table">
      <tr><td>Mean voxels</td><td>{df['voxels'].mean():.1f}</td></tr>
      <tr><td>Std deviation</td><td>{df['voxels'].std():.1f}</td></tr>
      <tr><td>Median voxels</td><td>{df['voxels'].median():.1f}</td></tr>
      <tr><td>Min voxels</td><td>{df['voxels'].min():.0f}</td></tr>
      <tr><td>Max voxels</td><td>{df['voxels'].max():.0f}</td></tr>
      <tr><td>25th percentile</td><td>{df['voxels'].quantile(0.25):.1f}</td></tr>
      <tr><td>75th percentile</td><td>{df['voxels'].quantile(0.75):.1f}</td></tr>
    </table>
  </div>
  <div class="card">
    <h2>Preprocessing Parameters</h2>
    <table class="kv-table">
      <tr><td>HU window min</td><td>-100 HU</td></tr>
      <tr><td>HU window max</td><td>900 HU</td></tr>
      <tr><td>Window center</td><td>400 HU</td></tr>
      <tr><td>Window width</td><td>1000 HU</td></tr>
      <tr><td>Target spacing XY</td><td>0.7 mm</td></tr>
      <tr><td>Target spacing Z</td><td>3.0 mm</td></tr>
      <tr><td>Padded/cropped shape</td><td>256×256×48</td></tr>
    </table>
  </div>
  <div class="card">
    <h2>Split Summary</h2>
    {"".join([f'''
    <div class="split-row">
      <span style="color:{'#6366f1' if s=='train' else '#22d3ee' if s=='val' else '#f472b6'}">{s}</span>
      <span>{split_counts[s]} scans</span>
      <div class="split-bar-wrap">
        <div class="split-bar" style="width:{100*split_counts[s]/len(df):.1f}%;background:{'#6366f1' if s=='train' else '#22d3ee' if s=='val' else '#f472b6'}"></div>
      </div>
      <span style="color:var(--muted)">{100*split_counts[s]/len(df):.1f}%</span>
    </div>''' for s in ['train','val','test']])}
    <div style="margin-top:1rem">
      <table class="kv-table">
        {"".join([f'<tr><td>{s} mean voxels</td><td>{df[df["split"]==s]["voxels"].mean():.1f}</td></tr>' for s in ["train","val","test"]])}
        {"".join([f'<tr><td>{s} zero%</td><td>{100*(df[df["split"]==s]["voxels"]==0).mean():.1f}%</td></tr>' for s in ["train","val","test"]])}
      </table>
    </div>
  </div>
</div>

<!-- Example Slice -->
{"" if not slice_b64 else f'''
<div class="card" style="margin-bottom:1.5rem">
  <h2>Example Axial Slice — {example_id} ({example_cat})</h2>
  <div class="img-pair" style="margin-top:1rem">
    <div class="img-wrap">
      <img src="data:image/png;base64,{slice_b64}" alt="CT slice"/>
      <p>Raw CT (HU windowed)</p>
    </div>
    <div class="img-wrap">
      <img src="data:image/png;base64,{overlay_b64}" alt="CT + calcium mask"/>
      <p>CT + calcium mask overlay (red)</p>
    </div>
  </div>
</div>
'''}

<script>
const catLabels  = {json.dumps(CAC_LABELS)};
const catCounts  = {json.dumps([cat_counts[c] for c in CAC_LABELS])};
const catColors  = {json.dumps(cat_colors)};
const histLabels = {json.dumps(hist_labels)};
const histCounts = {json.dumps(hist_counts)};
const splits     = ['train','val','test'];
const splitCounts= {json.dumps([split_counts[s] for s in ['train','val','test']])};
const splitColors= ['#6366f1','#22d3ee','#f472b6'];
const stackData  = {json.dumps(split_cat_data)};

const chartDefaults = {{
  plugins: {{ legend: {{ labels: {{ color: '#94a3b8', font: {{ family: 'JetBrains Mono', size: 11 }} }} }} }},
  scales: {{
    x: {{ ticks: {{ color: '#64748b', font: {{ family: 'JetBrains Mono', size: 10 }} }}, grid: {{ color: '#1e2d40' }} }},
    y: {{ ticks: {{ color: '#64748b', font: {{ family: 'JetBrains Mono', size: 10 }} }}, grid: {{ color: '#1e2d40' }} }}
  }}
}};

// CAC Category donut
new Chart(document.getElementById('catChart'), {{
  type: 'doughnut',
  data: {{ labels: catLabels, datasets: [{{ data: catCounts, backgroundColor: catColors, borderWidth: 0, hoverOffset: 6 }}] }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    cutout: '65%',
    plugins: {{ legend: {{ position: 'right', labels: {{ color: '#94a3b8', font: {{ family:'JetBrains Mono', size:11 }}, padding:12 }} }} }}
  }}
}});

// Split pie
new Chart(document.getElementById('splitChart'), {{
  type: 'doughnut',
  data: {{ labels: splits, datasets: [{{ data: splitCounts, backgroundColor: splitColors, borderWidth: 0, hoverOffset: 6 }}] }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    cutout: '65%',
    plugins: {{ legend: {{ position: 'right', labels: {{ color: '#94a3b8', font: {{ family:'JetBrains Mono', size:11 }}, padding:12 }} }} }}
  }}
}});

// Histogram
new Chart(document.getElementById('histChart'), {{
  type: 'bar',
  data: {{ labels: histLabels, datasets: [{{ label: 'Scans', data: histCounts, backgroundColor: '#38bdf8aa', borderColor: '#38bdf8', borderWidth: 1 }}] }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    ...chartDefaults,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{ ticks: {{ color:'#64748b', font:{{ family:'JetBrains Mono', size:9 }}, maxRotation:45 }}, grid:{{ color:'#1e2d40' }} }},
      y: {{ ticks: {{ color:'#64748b', font:{{ family:'JetBrains Mono', size:10 }} }}, grid:{{ color:'#1e2d40' }} }}
    }}
  }}
}});

// Stacked bar — proportions per split
new Chart(document.getElementById('stackChart'), {{
  type: 'bar',
  data: {{
    labels: splits,
    datasets: catLabels.map((cat, i) => ({{
      label: cat,
      data: splits.map(s => stackData[s][i]),
      backgroundColor: catColors[i],
      borderWidth: 0
    }}))
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    ...chartDefaults,
    scales: {{
      x: {{ stacked: true, ticks:{{ color:'#64748b', font:{{ family:'JetBrains Mono', size:11 }} }}, grid:{{ color:'#1e2d40' }} }},
      y: {{ stacked: true, ticks:{{ color:'#64748b', font:{{ family:'JetBrains Mono', size:11 }} }}, grid:{{ color:'#1e2d40' }} }}
    }}
  }}
}});
</script>
</body>
</html>"""

    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"\n✅ HTML report saved → {OUT_HTML}")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading split index...")
    df = pd.read_csv(SPLIT_CSV)

    print("Collecting spacing stats (sampling 80 scans)...")
    spacings = collect_spacing_stats(df)

    print("Collecting volume shapes (sampling 80 scans)...")
    shapes = collect_volume_shapes(df)

    print("Collecting intensity stats (sampling 30 scans)...")
    intensity_stats = collect_intensity_stats(df)

    print("Loading example slice...")
    img_slice, seg_slice, example_id, example_cat = get_example_slice(df)

    print_console_summary(df)

    print("\nBuilding HTML report...")
    build_html_report(df, spacings, shapes, intensity_stats,
                      img_slice, seg_slice, example_id, example_cat)
    print("\nBuilding PDF report...")
    save_pdf_report(df, intensity_stats, img_slice, seg_slice, example_id, example_cat)
