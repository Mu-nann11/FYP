"""
Visualization report generator — produces a self-contained HTML report
with interactive charts (embedded Chart.js) and summary tables.

No external dependencies beyond pandas/numpy (Chart.js loaded from CDN).
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np


def _fig_to_base64(fig_path: Path) -> str:
    """Read image file → base64 data URI for embedding in HTML."""
    import base64
    if not fig_path.exists():
        return ""
    with open(fig_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    ext = fig_path.suffix.lower().lstrip(".")
    mime = {"png": "image/png", "tif": "image/tiff", "tiff": "image/tiff",
            "jpg": "image/jpeg", "jpeg": "image/jpeg", "svg": "image/svg+xml"}
    return f"data:{mime.get(ext, 'image/png')};base64,{data}"


def _json_safe(val):
    """Make a value JSON-serializable."""
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        v = float(val)
        return v if np.isfinite(v) else None
    if isinstance(val, (np.bool_,)):
        return bool(val)
    if pd.isna(val):
        return None
    return val


def generate_report(
    cell_csv: Path,
    patient_csv: Optional[Path] = None,
    kappa_csv: Optional[Path] = None,
    overlay_dir: Optional[Path] = None,
    out_html: Optional[Path] = None,
    title: str = "TMA Analysis Report",
) -> Path:
    """
    Generate a self-contained HTML report.

    Parameters
    ----------
    cell_csv : path to all_blocks_cell_features CSV
    patient_csv : path to patient_level_scores CSV (optional)
    kappa_csv : path to kappa CSV (optional)
    overlay_dir : directory containing overlay images (optional)
    out_html : output HTML path (default: same dir as cell_csv, named report_*.html)
    title : report title

    Returns
    -------
    Path to generated HTML file
    """
    cell_csv = Path(cell_csv)
    df = pd.read_csv(cell_csv, encoding="utf-8-sig")
    out_html = out_html or cell_csv.parent / f"report_{cell_csv.stem}.html"

    # ---- Data preparation ----
    dataset = df["dataset"].iloc[0] if "dataset" in df.columns and len(df) else "Unknown"
    n_cells = len(df)
    n_blocks = df["block"].nunique() if "block" in df.columns else 0
    blocks = sorted(df["block"].unique().tolist()) if "block" in df.columns else []

    # Per-block summary
    block_stats = []
    for b in blocks:
        sub = df[df["block"] == b]
        row = {"block": b, "n_cells": len(sub)}
        for marker in ["ER", "PR"]:
            status_col = f"{marker}_status"
            frac_col = f"{marker}_positive_fraction"
            block_status_col = f"{marker}_block_status"
            if status_col in sub.columns:
                pos = (sub[status_col] == "Positive").sum()
                row[f"{marker}_positive"] = int(pos)
                row[f"{marker}_negative"] = int(len(sub) - pos)
                row[f"{marker}_frac"] = round(pos / len(sub), 4) if len(sub) else 0
            if block_status_col in sub.columns:
                row[f"{marker}_block_call"] = sub[block_status_col].iloc[0]
        # Ki67
        if "KI67_status" in sub.columns:
            pos = (sub["KI67_status"] == "Positive").sum()
            row["ki67_positive"] = int(pos)
            row["ki67_frac"] = round(pos / len(sub), 4) if len(sub) else 0
        if "ki67_proliferation_index" in sub.columns:
            row["ki67_index"] = _json_safe(sub["ki67_proliferation_index"].iloc[0])
        if "ki67_hotspot_proliferation_index" in sub.columns:
            row["ki67_hotspot_index"] = _json_safe(sub["ki67_hotspot_proliferation_index"].iloc[0])
        # HER2
        if "HER2_score" in sub.columns:
            her2_counts = sub["HER2_score"].value_counts().to_dict()
            row["her2_3plus"] = int(her2_counts.get("3+", 0))
            row["her2_2plus"] = int(her2_counts.get("2+", 0))
            row["her2_1plus"] = int(her2_counts.get("1+", 0))
            row["her2_0"] = int(her2_counts.get("0", 0))
        block_stats.append(row)

    # Patient-level
    patient_rows = []
    if patient_csv and Path(patient_csv).exists():
        pdf = pd.read_csv(patient_csv, encoding="utf-8-sig")
        patient_rows = pdf.to_dict(orient="records")

    # Kappa
    kappa_rows = []
    if kappa_csv and Path(kappa_csv).exists():
        kdf = pd.read_csv(kappa_csv, encoding="utf-8-sig")
        kappa_rows = kdf.to_dict(orient="records")

    # Overlay images
    overlay_images = []
    if overlay_dir and Path(overlay_dir).exists():
        for img in sorted(Path(overlay_dir).glob("*.png")):
            b64 = _fig_to_base64(img)
            if b64:
                overlay_images.append({"name": img.name, "src": b64})

    # ---- Build chart data ----
    # ER/PR per-block bar chart
    chart_labels = [s["block"] for s in block_stats]
    er_pos = [s.get("ER_positive", 0) for s in block_stats]
    er_neg = [s.get("ER_negative", 0) for s in block_stats]
    pr_pos = [s.get("PR_positive", 0) for s in block_stats]
    pr_neg = [s.get("PR_negative", 0) for s in block_stats]
    ki67_frac = [s.get("ki67_frac", 0) for s in block_stats]
    ki67_hotspot = [s.get("ki67_hotspot_index") for s in block_stats]

    # HER2 distribution
    her2_3p = [s.get("her2_3plus", 0) for s in block_stats]
    her2_2p = [s.get("her2_2plus", 0) for s in block_stats]
    her2_1p = [s.get("her2_1plus", 0) for s in block_stats]
    her2_0 = [s.get("her2_0", 0) for s in block_stats]

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  :root {{
    --bg: #0f1117;
    --card: #1a1d27;
    --border: #2a2d3a;
    --text: #e0e0e0;
    --text-dim: #888;
    --accent: #6c5ce7;
    --green: #00b894;
    --red: #e17055;
    --yellow: #fdcb6e;
    --blue: #0984e3;
    --cyan: #00cec9;
  }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
    padding: 2rem;
  }}
  .container {{ max-width: 1200px; margin: 0 auto; }}
  h1 {{
    font-size: 1.8rem;
    margin-bottom: 0.3rem;
    background: linear-gradient(135deg, var(--accent), var(--cyan));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }}
  .subtitle {{ color: var(--text-dim); margin-bottom: 2rem; font-size: 0.9rem; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 2rem; }}
  .stat-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
  }}
  .stat-card .number {{ font-size: 2rem; font-weight: 700; color: var(--cyan); }}
  .stat-card .label {{ font-size: 0.8rem; color: var(--text-dim); margin-top: 0.3rem; }}
  .section {{ margin-bottom: 2rem; }}
  .section-title {{
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
  }}
  .card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
  }}
  th {{
    text-align: left;
    padding: 0.6rem 0.8rem;
    background: rgba(108,92,231,0.15);
    border-bottom: 1px solid var(--border);
    font-weight: 600;
    white-space: nowrap;
  }}
  td {{
    padding: 0.5rem 0.8rem;
    border-bottom: 1px solid var(--border);
  }}
  tr:hover td {{ background: rgba(255,255,255,0.03); }}
  .badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 600;
  }}
  .badge-pos {{ background: rgba(0,184,148,0.2); color: var(--green); }}
  .badge-neg {{ background: rgba(225,112,85,0.2); color: var(--red); }}
  .badge-3 {{ background: rgba(0,184,148,0.25); color: var(--green); }}
  .badge-2 {{ background: rgba(253,203,110,0.25); color: var(--yellow); }}
  .badge-1 {{ background: rgba(116,185,255,0.2); color: var(--blue); }}
  .badge-0 {{ background: rgba(136,136,136,0.2); color: var(--text-dim); }}
  .chart-container {{ position: relative; height: 300px; margin: 1rem 0; }}
  .overlay-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 1rem;
  }}
  .overlay-item {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
  }}
  .overlay-item img {{
    width: 100%;
    height: auto;
    display: block;
    cursor: pointer;
  }}
  .overlay-item .caption {{
    padding: 0.5rem 0.8rem;
    font-size: 0.75rem;
    color: var(--text-dim);
    border-top: 1px solid var(--border);
  }}
  .lightbox {{
    display: none;
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background: rgba(0,0,0,0.9);
    z-index: 1000;
    justify-content: center;
    align-items: center;
    cursor: pointer;
  }}
  .lightbox.active {{ display: flex; }}
  .lightbox img {{ max-width: 95%; max-height: 95%; border-radius: 4px; }}
  .kappa-good {{ color: var(--green); font-weight: 700; }}
  .kappa-moderate {{ color: var(--yellow); font-weight: 700; }}
  .kappa-fair {{ color: var(--red); font-weight: 700; }}
  @media (max-width: 768px) {{
    body {{ padding: 1rem; }}
    .grid {{ grid-template-columns: repeat(2, 1fr); }}
  }}
</style>
</head>
<body>
<div class="container">

<h1>{title}</h1>
<p class="subtitle">Dataset: {dataset} &nbsp;|&nbsp; Generated: {now} &nbsp;|&nbsp; {n_cells:,} cells across {n_blocks} blocks</p>

<!-- Summary Stats -->
<div class="grid">
  <div class="stat-card">
    <div class="number">{n_cells:,}</div>
    <div class="label">Total Cells</div>
  </div>
  <div class="stat-card">
    <div class="number">{n_blocks}</div>
    <div class="label">Blocks</div>
  </div>
  <div class="stat-card">
    <div class="number">{len(patient_rows) // 2 if patient_rows else '—'}</div>
    <div class="label">Patients</div>
  </div>
  <div class="stat-card">
    <div class="number">{len(kappa_rows)}</div>
    <div class="label">Kappa Metrics</div>
  </div>
</div>

<!-- Block-level ER/PR -->
<div class="section">
  <div class="section-title">📊 Block-level ER / PR Status</div>
  <div class="card">
    <div class="chart-container">
      <canvas id="erprChart"></canvas>
    </div>
    <table>
      <thead>
        <tr>
          <th>Block</th><th>Cells</th>
          <th>ER+</th><th>ER−</th><th>ER %</th><th>ER Call</th>
          <th>PR+</th><th>PR−</th><th>PR %</th><th>PR Call</th>
        </tr>
      </thead>
      <tbody>
"""
    for s in block_stats:
        er_pct = f"{s.get('ER_frac', 0)*100:.1f}%"
        pr_pct = f"{s.get('PR_frac', 0)*100:.1f}%"
        er_call = s.get("ER_block_call", "—")
        pr_call = s.get("PR_block_call", "—")
        er_badge = 'badge-pos' if er_call == 'Positive' else 'badge-neg'
        pr_badge = 'badge-pos' if pr_call == 'Positive' else 'badge-neg'
        html += f"""        <tr>
          <td><strong>{s['block']}</strong></td><td>{s['n_cells']:,}</td>
          <td>{s.get('ER_positive',0)}</td><td>{s.get('ER_negative',0)}</td>
          <td>{er_pct}</td><td><span class="badge {er_badge}">{er_call}</span></td>
          <td>{s.get('PR_positive',0)}</td><td>{s.get('PR_negative',0)}</td>
          <td>{pr_pct}</td><td><span class="badge {pr_badge}">{pr_call}</span></td>
        </tr>
"""

    html += """      </tbody>
    </table>
  </div>
</div>

<!-- Ki67 -->
<div class="section">
  <div class="section-title">🔥 Ki67 Proliferation Index</div>
  <div class="card">
    <div class="chart-container">
      <canvas id="ki67Chart"></canvas>
    </div>
    <table>
      <thead>
        <tr><th>Block</th><th>Ki67+</th><th>Ki67−</th><th>Index %</th><th>Hotspot Index %</th></tr>
      </thead>
      <tbody>
"""
    for s in block_stats:
        if "ki67_positive" in s or "ki67_index" in s:
            html += f"""        <tr>
          <td><strong>{s['block']}</strong></td>
          <td>{s.get('ki67_positive', '—')}</td>
          <td>{s.get('n_cells',0) - s.get('ki67_positive',0)}</td>
          <td>{s.get('ki67_index', '—')}</td>
          <td>{s.get('ki67_hotspot_index', '—')}</td>
        </tr>
"""
    html += """      </tbody>
    </table>
  </div>
</div>

<!-- HER2 -->
<div class="section">
  <div class="section-title">🧬 HER2 Score Distribution</div>
  <div class="card">
    <div class="chart-container">
      <canvas id="her2Chart"></canvas>
    </div>
  </div>
</div>
"""

    # Patient-level
    if patient_rows:
        html += """
<!-- Patient-level -->
<div class="section">
  <div class="section-title">🏥 Patient-level ER / PR Summary</div>
  <div class="card">
    <table>
      <thead>
        <tr><th>Patient</th><th>Marker</th><th>Total Cells</th><th>Positive</th><th>Fraction</th><th>Status</th><th>Blocks</th></tr>
      </thead>
      <tbody>
"""
        for r in patient_rows:
            status = r.get("patient_status", "—")
            badge = 'badge-pos' if status == 'Positive' else 'badge-neg'
            frac = r.get("positive_fraction", 0)
            html += f"""        <tr>
          <td><strong>{r.get('patient_id','')}</strong></td>
          <td>{r.get('marker','')}</td>
          <td>{r.get('total_cells',0):,}</td>
          <td>{r.get('n_positive',0)}</td>
          <td>{frac*100:.2f}%</td>
          <td><span class="badge {badge}">{status}</span></td>
          <td>{r.get('blocks','')}</td>
        </tr>
"""
        html += """      </tbody>
    </table>
  </div>
</div>
"""

    # Kappa
    if kappa_rows:
        html += """
<!-- Kappa -->
<div class="section">
  <div class="section-title">📐 Cohen's Kappa (vs QuPath)</div>
  <div class="card">
    <table>
      <thead>
        <tr><th>Marker</th><th>Kappa (κ)</th><th>Interpretation</th><th>Agreement Rate</th><th>N Samples</th></tr>
      </thead>
      <tbody>
"""
        for r in kappa_rows:
            k = r.get("kappa", float("nan"))
            if k is None or (isinstance(k, float) and np.isnan(k)):
                interp, cls = "N/A", ""
            elif k >= 0.8:
                interp, cls = "Almost perfect", "kappa-good"
            elif k >= 0.6:
                interp, cls = "Substantial", "kappa-good"
            elif k >= 0.4:
                interp, cls = "Moderate", "kappa-moderate"
            elif k >= 0.2:
                interp, cls = "Fair", "kappa-fair"
            else:
                interp, cls = "Slight / Poor", "kappa-fair"
            n = r.get("n_samples", r.get("n_patients", "—"))
            html += f"""        <tr>
          <td><strong>{r.get('marker','')}</strong></td>
          <td class="{cls}">{k if k is not None else '—'}</td>
          <td>{interp}</td>
          <td>{r.get('agreement_rate', '—')}</td>
          <td>{n}</td>
        </tr>
"""
        html += """      </tbody>
    </table>
  </div>
</div>
"""

    # Overlay gallery
    if overlay_images:
        html += """
<!-- Overlays -->
<div class="section">
  <div class="section-title">🖼️ Overlay Visualizations</div>
  <div class="card">
    <div class="overlay-grid">
"""
        for img in overlay_images:
            html += f"""      <div class="overlay-item">
        <img src="{img['src']}" onclick="openLightbox(this.src)" alt="{img['name']}">
        <div class="caption">{img['name']}</div>
      </div>
"""
        html += """    </div>
  </div>
</div>
"""

    # Lightbox
    html += """
<div class="lightbox" id="lightbox" onclick="this.classList.remove('active')">
  <img id="lightboxImg" src="">
</div>
</div>

<script>
function openLightbox(src) {
  document.getElementById('lightboxImg').src = src;
  document.getElementById('lightbox').classList.add('active');
}

// ER/PR Chart
new Chart(document.getElementById('erprChart'), {
  type: 'bar',
  data: {
    labels: """ + json.dumps(chart_labels) + """,
    datasets: [
      { label: 'ER+', data: """ + json.dumps(er_pos) + """, backgroundColor: 'rgba(0,184,148,0.7)' },
      { label: 'ER−', data: """ + json.dumps(er_neg) + """, backgroundColor: 'rgba(0,184,148,0.2)' },
      { label: 'PR+', data: """ + json.dumps(pr_pos) + """, backgroundColor: 'rgba(9,132,227,0.7)' },
      { label: 'PR−', data: """ + json.dumps(pr_neg) + """, backgroundColor: 'rgba(9,132,227,0.2)' },
    ]
  },
  options: {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { labels: { color: '#ccc' } } },
    scales: {
      x: { stacked: true, ticks: { color: '#aaa' }, grid: { color: 'rgba(255,255,255,0.05)' } },
      y: { stacked: true, ticks: { color: '#aaa' }, grid: { color: 'rgba(255,255,255,0.05)' } }
    }
  }
});

// Ki67 Chart
new Chart(document.getElementById('ki67Chart'), {
  type: 'bar',
  data: {
    labels: """ + json.dumps(chart_labels) + """,
    datasets: [
      { label: 'Ki67 Index %', data: """ + json.dumps([x if x is not None else 0 for x in ki67_frac], default=str).replace('"null"', 'null') + """,
        backgroundColor: 'rgba(225,112,85,0.6)', yAxisID: 'y' },
      { label: 'Hotspot Index %', data: """ + json.dumps([x if x is not None else 0 for x in ki67_hotspot], default=str).replace('"null"', 'null') + """,
        backgroundColor: 'rgba(253,203,110,0.6)', yAxisID: 'y' },
    ]
  },
  options: {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { labels: { color: '#ccc' } } },
    scales: {
      x: { ticks: { color: '#aaa' }, grid: { color: 'rgba(255,255,255,0.05)' } },
      y: { beginAtZero: true, max: 100, ticks: { color: '#aaa', callback: v => v + '%' },
           grid: { color: 'rgba(255,255,255,0.05)' } }
    }
  }
});

// HER2 Chart
new Chart(document.getElementById('her2Chart'), {
  type: 'bar',
  data: {
    labels: """ + json.dumps(chart_labels) + """,
    datasets: [
      { label: '3+', data: """ + json.dumps(her2_3p) + """, backgroundColor: 'rgba(0,184,148,0.7)' },
      { label: '2+', data: """ + json.dumps(her2_2p) + """, backgroundColor: 'rgba(253,203,110,0.7)' },
      { label: '1+', data: """ + json.dumps(her2_1p) + """, backgroundColor: 'rgba(9,132,227,0.7)' },
      { label: '0',  data: """ + json.dumps(her2_0) + """, backgroundColor: 'rgba(136,136,136,0.5)' },
    ]
  },
  options: {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { labels: { color: '#ccc' } } },
    scales: {
      x: { stacked: true, ticks: { color: '#aaa' }, grid: { color: 'rgba(255,255,255,0.05)' } },
      y: { stacked: true, ticks: { color: '#aaa' }, grid: { color: 'rgba(255,255,255,0.05)' } }
    }
  }
});
</script>
</body>
</html>"""

    out_html.write_text(html, encoding="utf-8")
    return out_html


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate TMA HTML report")
    parser.add_argument("--cell-csv", required=True, help="Path to all_blocks_cell_features CSV")
    parser.add_argument("--patient-csv", default=None, help="Path to patient_level_scores CSV")
    parser.add_argument("--kappa-csv", default=None, help="Path to kappa CSV")
    parser.add_argument("--overlay-dir", default=None, help="Path to overlay images dir")
    parser.add_argument("--out", default=None, help="Output HTML path")
    parser.add_argument("--title", default="TMA Analysis Report")
    args = parser.parse_args()

    out = generate_report(
        cell_csv=args.cell_csv,
        patient_csv=args.patient_csv,
        kappa_csv=args.kappa_csv,
        overlay_dir=args.overlay_dir,
        out_html=Path(args.out) if args.out else None,
        title=args.title,
    )
    print(f"Report generated: {out}")
