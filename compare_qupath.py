from pathlib import Path
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

from fiji_stitcher.config import load_config


def compare_qupath(
    cp_csv: str | Path,
    qp_txt: str | Path,
    out_csv: str | Path,
    max_dist: float = 15.0,
):
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
        "cp_area": matched_cp["area"].astype(float),
        "distance": distances,
        "matched_within_max_dist": valid,
    })

    result_valid = result[valid].copy()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    result_valid.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("Cellpose 细胞数 (ROI 内):", len(df_cp))
    print("QuPath 细胞数 (ROI 内):", len(df_qp))
    print(f"匹配率: {valid.sum() / len(result) * 100.0:.1f}%")
    print("距离中位数 / 平均值:", np.median(distances), np.mean(distances))
    print("面积比中位数:", np.median(result["cp_area"] / result["qp_area"]))
    print("已保存匹配结果到:", out_csv)


if __name__ == "__main__":
    config = load_config()
    compare_cfg = config.get("COMPARE_QUPATH", {})

    cp_csv = compare_cfg.get("CELLPOSE_CSV", "A4_cell_features_roi_only.csv")
    qp_txt = compare_cfg.get("QUPATH_TXT", "A4_DAPI_qupath_cells_manual.txt")
    out_csv = compare_cfg.get("OUT_CSV", "A4_qupath_vs_cellpose_matches.csv")
    max_dist = float(compare_cfg.get("MAX_DIST", 15.0))

    compare_qupath(cp_csv=cp_csv, qp_txt=qp_txt, out_csv=out_csv, max_dist=max_dist)
