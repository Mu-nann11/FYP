from pathlib import Path
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree


def compare_qupath(
    cp_csv: str | Path,
    qp_txt: str | Path,
    out_csv: str | Path,
    max_dist: float = 15.0,
    marker_cols: list = None,
):
    """
    Compare Cellpose pipeline results against QuPath reference.

    Matches cells by nearest-centroid spatial proximity, then outputs
    matched pairs with both sets of marker statuses for Cohen's kappa.

    Parameters
    ----------
    cp_csv : path to our pipeline CSV (must have centroid_x, centroid_y)
    qp_txt : path to QuPath export (TSV, must have Centroid X µm, Centroid Y µm)
    out_csv : output matched CSV
    max_dist : max pixel distance for a valid match
    marker_cols : list of marker status column names from our CSV
                  (e.g. ["ER_status", "PR_status", "HER2_score", "KI67_status"])
                  If None, auto-detected from columns ending with _status or _score
    """
    cp_csv = Path(cp_csv)
    qp_txt = Path(qp_txt)
    out_csv = Path(out_csv)

    df_cp = pd.read_csv(cp_csv)
    df_qp = pd.read_csv(qp_txt, sep="\t", header=0)

    cx_cp, cy_cp = "centroid_x", "centroid_y"
    cx_qp, cy_qp = "Centroid X µm", "Centroid Y µm"
    qp_area_col = "Area µm^2"

    cp_points = df_cp[[cx_cp, cy_cp]].to_numpy()
    qp_points = df_qp[[cx_qp, cy_qp]].to_numpy()

    tree = cKDTree(cp_points)
    distances, indices = tree.query(qp_points, k=1)
    valid = distances <= max_dist

    matched_cp = df_cp.iloc[indices].reset_index(drop=True)

    result = pd.DataFrame({
        "qp_object_id": df_qp["Object ID"],
        "qp_centroid_x": df_qp[cx_qp],
        "qp_centroid_y": df_qp[cy_qp],
        "qp_area": df_qp[qp_area_col].astype(float),
        "cp_cell_id": matched_cp["cell_id"],
        "cp_centroid_x": matched_cp[cx_cp],
        "cp_centroid_y": matched_cp[cy_cp],
        "cp_area": matched_cp["nuc_area"].astype(float) if "nuc_area" in matched_cp.columns else matched_cp.get("area", 0).astype(float),
        "distance": distances,
        "matched_within_max_dist": valid,
    })

    # Include global_cell_id if available (for kappa merging)
    if "global_cell_id" in matched_cp.columns:
        result["global_cell_id"] = matched_cp["global_cell_id"].values

    # Auto-detect or use specified marker columns
    if marker_cols is None:
        marker_cols = [c for c in matched_cp.columns
                       if c.endswith("_status") or c.endswith("_score")]

    # Add our pipeline marker columns
    for col in marker_cols:
        if col in matched_cp.columns:
            result[f"cp_{col}"] = matched_cp[col].values

    # Try to extract QuPath classifications (common QuPath column names)
    qp_class_col = "Classification" if "Classification" in df_qp.columns else None
    if qp_class_col:
        result["qp_classification"] = df_qp[qp_class_col].values

    # Try QuPath intensity columns for marker scoring
    for col in df_qp.columns:
        col_lower = col.lower()
        for marker in ["er", "pr", "her2", "ki67"]:
            if marker in col_lower and ("mean" in col_lower or "intensity" in col_lower):
                result[f"qp_{col}"] = df_qp[col].values
                break

    result_valid = result[valid].copy()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    result_valid.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("Cellpose 细胞数 (ROI 内):", len(df_cp))
    print("QuPath 细胞数 (ROI 内):", len(df_qp))
    print(f"匹配率: {valid.sum() / len(result) * 100.0:.1f}%")
    print("距离中位数 / 平均值:", np.median(distances), np.mean(distances))
    print("面积比中位数:", np.median(result["cp_area"] / result["qp_area"]))
    print("Marker columns included:", marker_cols)
    print("已保存匹配结果到:", out_csv)


if __name__ == "__main__":
    # Standalone config import (uses fiji_config.json in same dir)
    import json
    config_path = Path(__file__).parent / "fiji_config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {}

    compare_cfg = config.get("COMPARE_QUPATH", {})

    cp_csv = compare_cfg.get("CELLPOSE_CSV", "A4_cell_features_roi_only.csv")
    qp_txt = compare_cfg.get("QUPATH_TXT", "A4_DAPI_qupath_cells_manual.txt")
    out_csv = compare_cfg.get("OUT_CSV", "A4_qupath_vs_cellpose_matches.csv")
    max_dist = float(compare_cfg.get("MAX_DIST", 15.0))

    compare_qupath(cp_csv=cp_csv, qp_txt=qp_txt, out_csv=out_csv, max_dist=max_dist)
