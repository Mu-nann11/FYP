import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import sys
import time

from loader import load_block
from segmentation import segment_nuclei, get_cytoplasm_masks
from utils import q90


def _progress(prefix: str, current: int, total: int):
    if total <= 0:
        return
    pct = int(current * 100 / total)
    msg = f"\r{prefix} {current}/{total} ({pct}%)"
    sys.stdout.write(msg)
    sys.stdout.flush()


def _progress_done():
    sys.stdout.write("\n")
    sys.stdout.flush()


def compute_block(block: str, expansion_px: int) -> pd.DataFrame:
    t0 = time.time()
    print(f"[{block} | {expansion_px}px] loading images...")
    imgs = load_block(block, do_preprocess=True)
    dapi = imgs["DAPI"]
    her2 = imgs["HER2"]

    print(f"[{block} | {expansion_px}px] segmenting nuclei...")
    nuclei = segment_nuclei(dapi)
    _, cyto_only = get_cytoplasm_masks(nuclei, expansion_distance=expansion_px)

    rows = []
    max_id = int(nuclei.max())
    step = max(1, max_id // 50)
    print(f"[{block} | {expansion_px}px] extracting ring metrics...")
    for cid in range(1, max_id + 1):
        if cid == 1 or cid == max_id or cid % step == 0:
            _progress(f"[{block} | {expansion_px}px] cells", cid, max_id)
        nuc_mask = nuclei == cid
        ring_mask = cyto_only == cid

        nuc_area = int(nuc_mask.sum())
        ring_area = int(ring_mask.sum())
        if nuc_area == 0:
            continue

        nuc_vals = her2[nuc_mask].astype(np.float64)
        if ring_area > 0:
            ring_vals = her2[ring_mask].astype(np.float64)
            ring_median = float(np.median(ring_vals))
            ring_p90 = q90(ring_vals)
        else:
            ring_median = float("nan")
            ring_p90 = float("nan")

        rows.append(
            {
                "block": block,
                "cell_id": cid,
                "nuc_area_px": nuc_area,
                "ring_area_px": ring_area,
                "her2_nuc_mean": float(nuc_vals.mean()),
                "her2_nuc_p90": q90(nuc_vals),
                "her2_ring_median": ring_median,
                "her2_ring_p90": ring_p90,
            }
        )

    _progress_done()
    print(f"[{block} | {expansion_px}px] done in {time.time() - t0:.1f}s, cells={len(rows)}")
    return pd.DataFrame(rows)


def summarize_cells(df_cells: pd.DataFrame) -> pd.DataFrame:
    if len(df_cells) == 0:
        return pd.DataFrame(columns=["block"])

    return (
        df_cells.groupby("block")
        .agg(
            n_cells=("cell_id", "count"),
            her2_ring_median_median=("her2_ring_median", "median"),
            her2_ring_p90_median=("her2_ring_p90", "median"),
            her2_ring_p90_p90=("her2_ring_p90", q90),
        )
        .reset_index()
        .sort_values("block")
        .reset_index(drop=True)
    )


def load_excel_labels(blocks: list[str]) -> pd.DataFrame:
    excel_path = Path("1_BC081120e specs.xlsx")
    df_excel = pd.read_excel(excel_path, skiprows=10)
    df_excel.columns = [str(c).strip() for c in df_excel.columns]
    labels = df_excel[df_excel["Position"].astype(str).isin(blocks)][["Position", "HER2"]].copy()
    labels = labels.rename(columns={"Position": "block", "HER2": "HER2_label"}).sort_values("block")
    return labels.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks", nargs="+", default=["F10", "E5"])
    parser.add_argument("--expansion", nargs="+", type=int, default=[10, 15, 20])
    args = parser.parse_args()

    blocks = [str(b).strip() for b in args.blocks if str(b).strip()]
    expansions = [int(x) for x in args.expansion]

    out_dir = Path("/results/compared_result")
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = load_excel_labels(blocks)
    labels.to_csv(out_dir / "ring_metrics_excel_labels.csv", index=False, encoding="utf-8-sig")

    all_reports = []
    for expansion_px in expansions:
        dfs = [compute_block(b, expansion_px=expansion_px) for b in blocks]
        df_cells = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

        cells_name = "ring_metrics_cells_%s_px.csv" % expansion_px
        df_cells.to_csv(out_dir / cells_name, index=False, encoding="utf-8-sig")

        if len(df_cells) == 0:
            continue

        summary = summarize_cells(df_cells)
        summary_name = "ring_metrics_summary_%s_px.csv" % expansion_px
        summary.to_csv(out_dir / summary_name, index=False, encoding="utf-8-sig")

        merged = labels.merge(summary, on="block", how="left")
        merged["expansion_px"] = expansion_px
        all_reports.append(merged)

    if not all_reports:
        raise RuntimeError("no reports")

    report = pd.concat(all_reports, ignore_index=True)
    report.to_csv(out_dir / "ring_metrics_excel_vs_ring_multi_expansion.csv", index=False, encoding="utf-8-sig")

    show_cols = ["expansion_px", "block", "HER2_label", "n_cells", "her2_ring_median_median", "her2_ring_p90_median", "her2_ring_p90_p90"]
    print(report[show_cols].sort_values(["expansion_px", "block"]).to_string(index=False))


if __name__ == "__main__":
    main()
