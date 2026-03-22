import numpy as np
import pandas as pd
from skimage.measure import regionprops
from skimage.filters import threshold_otsu
import cv2
from utils import q90
from config import config


def _safe_otsu_threshold(vals: np.ndarray, fallback: float) -> float:
    """Otsu on 1D intensities; fall back if degenerate."""
    v = np.asarray(vals, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return float(fallback)
    vmin, vmax = float(v.min()), float(v.max())
    if vmax <= vmin:
        return float(fallback)
    try:
        return float(threshold_otsu(v))
    except (ValueError, RuntimeError):
        return float(np.median(v))


def _greedy_intensity_seeds(
    df: pd.DataFrame,
    value_col: str,
    n_seeds: int,
    min_sep_px: float,
) -> list[tuple[float, float]]:
    sub = df[["centroid_x", "centroid_y", value_col]].dropna()
    if sub.empty or n_seeds < 1:
        return []
    sub = sub.sort_values(value_col, ascending=False)
    picked: list[tuple[float, float]] = []
    for _, row in sub.iterrows():
        cx, cy = float(row["centroid_x"]), float(row["centroid_y"])
        if all(np.hypot(cx - px, cy - py) >= min_sep_px for px, py in picked):
            picked.append((cx, cy))
            if len(picked) >= n_seeds:
                break
    return picked


def _hotspot_mask_from_seeds(df: pd.DataFrame, seeds: list[tuple[float, float]], radius_px: float) -> pd.Series:
    if not seeds or radius_px <= 0:
        return pd.Series(False, index=df.index)
    xy = df[["centroid_x", "centroid_y"]]
    mask = np.zeros(len(df), dtype=bool)
    arr = xy.to_numpy(dtype=np.float64)
    valid = np.isfinite(arr).all(axis=1)
    for sx, sy in seeds:
        d = np.linalg.norm(arr - np.array([sx, sy], dtype=np.float64), axis=1)
        mask |= valid & (d <= radius_px)
    return pd.Series(mask, index=df.index)

def extract_features(block_name, nuclei_masks, cyto_only_masks, channels_dict, cell_masks=None):
    """
    优化版特征提取：
    1. 使用 regionprops_table 快速提取形态学特征。
    2. 使用 regionprops(intensity_image=...) 批量获取通道强度，减少循环内坐标索引。
    3. 测量核膜距离。
    """
    # 1. 核形态学基础特征
    props_nuc = regionprops(nuclei_masks)
    if not props_nuc:
        return pd.DataFrame()

    # 2. 准备结果列表
    cid_list = [p.label for p in props_nuc]
    n_cells = len(cid_list)
    
    # 建立映射方便通过 label 获取索引
    label_to_idx = {label: i for i, label in enumerate(cid_list)}

    # 初始化 DataFrame
    df = pd.DataFrame({
        "block": [block_name] * n_cells,
        "cell_id": cid_list,
        "global_cell_id": ["%s_%s" % (block_name, cid) for cid in cid_list],
        "nuc_area": [float(p.area) for p in props_nuc],
        "nuc_eccentricity": [float(p.eccentricity) for p in props_nuc],
        "centroid_y": [float(p.centroid[0]) for p in props_nuc],
        "centroid_x": [float(p.centroid[1]) for p in props_nuc],
    })

    # 3. 胞质面积 (从 cell_masks 或 cyto_only_masks 获取)
    if cell_masks is not None:
        cell_props = regionprops(cell_masks)
        cell_area_map = {p.label: float(p.area) for p in cell_props}
        df["cell_area"] = df["cell_id"].map(lambda x: cell_area_map.get(x, 0.0))
    else:
        df["cell_area"] = df["nuc_area"] # 如果没有胞质，则细胞面积=核面积

    cyto_props = regionprops(cyto_only_masks)
    cyto_area_map = {p.label: float(p.area) for p in cyto_props}
    df["cyto_area"] = df["cell_id"].map(lambda x: cyto_area_map.get(x, 0.0))
    df["cell_to_nuclear_ratio"] = df["cell_area"] / df.get("nuc_area", 1.0)

    # 4. 核膜距离计算 (仅对有胞质的细胞)
    inv_nuclei = (nuclei_masks == 0).astype(np.uint8)
    dist_map = cv2.distanceTransform(inv_nuclei, cv2.DIST_L2, 3).astype(np.float64)
    
    dist_means, dist_p90s, dist_maxs = [np.nan]*n_cells, [np.nan]*n_cells, [np.nan]*n_cells
    
    for p_cyto in cyto_props:
        idx = label_to_idx.get(p_cyto.label)
        if idx is not None:
            # 使用 coords 索引比直接在整个 map 上 masking 快一些，因为 cyto_only_masks 可能很大
            dists = dist_map[p_cyto.coords[:, 0], p_cyto.coords[:, 1]]
            if dists.size:
                dist_means[idx] = float(dists.mean())
                dist_p90s[idx] = q90(dists)
                dist_maxs[idx] = float(dists.max())
    
    df["cyto_to_nuc_dist_mean_px"] = dist_means
    df["cyto_to_nuc_dist_p90_px"] = dist_p90s
    df["cyto_to_nuc_dist_max_px"] = dist_maxs

    # 5. 通道强度提取 (核心优化：利用 regionprops 的 intensity_image)
    for ch_name, img in channels_dict.items():
        # 核内强度
        props_nuc_int = regionprops(nuclei_masks, intensity_image=img)
        nuc_means = {p.label: float(p.mean_intensity) for p in props_nuc_int}
        nuc_maxs = {p.label: float(p.max_intensity) for p in props_nuc_int}
        # q90 还是得手动算一下，因为 regionprops 不自带 q90
        nuc_p90s = {p.label: q90(p.intensity_image[p.image]) for p in props_nuc_int}

        df["%s_nuc_mean" % ch_name] = df["cell_id"].map(nuc_means)
        df["%s_nuc_max" % ch_name] = df["cell_id"].map(nuc_maxs)
        df["%s_nuc_p90" % ch_name] = df["cell_id"].map(nuc_p90s)

        # 胞质内强度
        props_cyto_int = regionprops(cyto_only_masks, intensity_image=img)
        cyto_means = {p.label: float(p.mean_intensity) for p in props_cyto_int}
        cyto_maxs = {p.label: float(p.max_intensity) for p in props_cyto_int}
        cyto_p90s = {p.label: q90(p.intensity_image[p.image]) for p in props_cyto_int}

        df["%s_cyto_mean" % ch_name] = df["cell_id"].map(lambda x: cyto_means.get(x, 0.0))
        df["%s_cyto_max" % ch_name] = df["cell_id"].map(lambda x: cyto_maxs.get(x, 0.0))
        df["%s_cyto_p90" % ch_name] = df["cell_id"].map(lambda x: cyto_p90s.get(x, np.nan))

        # HER2：核周薄环（膜区）强度，避免全胞质均值稀释膜信号
        if ch_name == "HER2":
            ring_lo = float(config.get("SCORING.HER2_MEMBRANE_RING_INNER_PX", 1))
            ring_hi = float(config.get("SCORING.HER2_MEMBRANE_RING_OUTER_PX", 4))
            mem_means: dict = {}
            mem_p90s: dict = {}
            for p_cyto in cyto_props:
                lab = p_cyto.label
                rr, cc = p_cyto.coords[:, 0], p_cyto.coords[:, 1]
                d = dist_map[rr, cc]
                in_ring = (d >= ring_lo) & (d <= ring_hi)
                if in_ring.any():
                    vals = img[rr[in_ring], cc[in_ring]].astype(np.float64, copy=False)
                    mem_means[lab] = float(vals.mean())
                    mem_p90s[lab] = q90(vals)
                else:
                    mem_means[lab] = float("nan")
                    mem_p90s[lab] = float("nan")
            df["HER2_membrane_mean"] = df["cell_id"].map(mem_means)
            df["HER2_membrane_p90"] = df["cell_id"].map(mem_p90s)

    return df

def score_markers(df):
    """
    免疫组化自动评分（块级 ER/PR 比例、Ki67 hotspot、HER2 膜区优先）。
    配置键见 fiji_config.json SCORING。
    """
    if df.empty:
        return df

    pos_thr = float(config.get("SCORING.POSITIVE_THRESHOLD", 5000))
    her2_thrs = config.get("SCORING.HER2_THRESHOLDS", {"3+": 15000, "2+": 8000, "1+": 3000})
    er_pr_mode = str(config.get("SCORING.ER_PR_THRESHOLD_MODE", "otsu")).strip().lower()
    min_pos_frac = float(config.get("SCORING.ER_PR_MIN_POSITIVE_FRACTION", 0.01))
    ki67_mode = str(config.get("SCORING.KI67_THRESHOLD_MODE", "hotspot_otsu")).strip().lower()
    ki67_n_seeds = int(config.get("SCORING.KI67_HOTSPOT_N_SEEDS", 3))
    ki67_min_sep = float(config.get("SCORING.KI67_HOTSPOT_MIN_SEP_PX", 20.0))
    ki67_radius = float(config.get("SCORING.KI67_HOTSPOT_RADIUS_PX", 150.0))
    ki67_min_hotspot_cells = int(config.get("SCORING.KI67_HOTSPOT_MIN_CELLS", 10))
    use_her2_ring = config.get("SCORING.HER2_USE_MEMBRANE_RING", True)
    her2_fallback_cyto = config.get("SCORING.HER2_FALLBACK_TO_CYTO_MEAN", True)

    # ---------- ER / PR：块内阈值 + 阳性比例 + 块级阴阳性 ----------
    for marker in ["ER", "PR"]:
        col = "%s_nuc_mean" % marker
        if col not in df.columns:
            continue
        vals = df[col].dropna().values
        if er_pr_mode in ("fixed", "legacy"):
            thr = pos_thr
        else:
            thr = _safe_otsu_threshold(vals, pos_thr)
        pos = df[col] > thr
        df["%s_nuc_positive" % marker] = pos
        frac = float(pos.mean()) if len(df) else 0.0
        df["%s_positive_fraction" % marker] = frac
        df["%s_threshold_used" % marker] = thr
        block_call = "Positive" if frac >= min_pos_frac else "Negative"
        df["%s_block_status" % marker] = block_call
        # 与旧列兼容：细胞级阴阳性 = 是否超过块内所用阈值
        df["%s_status" % marker] = np.where(pos, "Positive", "Negative")

    # ---------- HER2：优先膜区均值，缺失时可选回退胞质 ----------
    her2_col = None
    if use_her2_ring and "HER2_membrane_mean" in df.columns:
        m = df["HER2_membrane_mean"]
        if her2_fallback_cyto and "HER2_cyto_mean" in df.columns:
            her2_vals = m.fillna(df["HER2_cyto_mean"])
        else:
            her2_vals = m
        her2_col = her2_vals
    elif "HER2_cyto_mean" in df.columns:
        her2_col = df["HER2_cyto_mean"]

    if her2_col is not None:
        def score_her2(val):
            if not np.isfinite(val):
                return "0"
            if val > her2_thrs.get("3+", 15000):
                return "3+"
            if val > her2_thrs.get("2+", 8000):
                return "2+"
            if val > her2_thrs.get("1+", 3000):
                return "1+"
            return "0"

        df["HER2_score"] = her2_col.apply(score_her2)
        if "HER2_membrane_mean" in df.columns and use_her2_ring:
            df["HER2_score_basis"] = np.where(
                df["HER2_membrane_mean"].notna(),
                "membrane_ring",
                "cyto_mean",
            )
        else:
            df["HER2_score_basis"] = "cyto_mean"

    # ---------- Ki67：hotspot 内 Otsu 阈值；否则全视野 Otsu / 固定阈值 ----------
    if "KI67_nuc_mean" in df.columns:
        fixed_thr = config.get("SCORING.KI67_THRESHOLD", None)
        if fixed_thr is not None:
            thr = float(fixed_thr)
            df["KI67_in_hotspot"] = True
            df["KI67_threshold_used"] = thr
            df["KI67_threshold_mode_used"] = "fixed"
        elif ki67_mode in ("global_otsu", "global", "legacy"):
            vals = df["KI67_nuc_mean"].dropna().values
            thr = _safe_otsu_threshold(vals, pos_thr)
            df["KI67_in_hotspot"] = True
            df["KI67_threshold_used"] = thr
            df["KI67_threshold_mode_used"] = "global_otsu"
        else:
            seeds = _greedy_intensity_seeds(df, "KI67_nuc_mean", ki67_n_seeds, ki67_min_sep)
            hmask = _hotspot_mask_from_seeds(df, seeds, ki67_radius)
            df["KI67_in_hotspot"] = hmask
            hot_vals = df.loc[hmask, "KI67_nuc_mean"].dropna().values
            if hot_vals.size >= ki67_min_hotspot_cells:
                thr = _safe_otsu_threshold(hot_vals, pos_thr)
                df["KI67_threshold_mode_used"] = "hotspot_otsu"
            else:
                vals = df["KI67_nuc_mean"].dropna().values
                thr = _safe_otsu_threshold(vals, pos_thr)
                df["KI67_threshold_mode_used"] = "global_otsu_fallback"
                df["KI67_in_hotspot"] = True
            df["KI67_threshold_used"] = thr

        df["KI67_status"] = np.where(df["KI67_nuc_mean"] > thr, "Positive", "Negative")

    return df


def compute_ki67_index(df: pd.DataFrame) -> float:
    """
    全视野 Ki67 指数 = 阳性细胞数 / 总细胞数 × 100%（阈值由 score_markers 决定）。
    """
    if "KI67_status" not in df.columns or df.empty:
        return float("nan")
    n_pos = (df["KI67_status"] == "Positive").sum()
    return round(100.0 * n_pos / len(df), 2)


def compute_ki67_hotspot_index(df: pd.DataFrame) -> float:
    """
    Hotspot 内 Ki67 指数 = hotspot 区域内阳性 / hotspot 内细胞数 × 100%。
    若未标记 hotspot（全 True），与 compute_ki67_index 一致。
    """
    if "KI67_status" not in df.columns or df.empty:
        return float("nan")
    if "KI67_in_hotspot" not in df.columns:
        return compute_ki67_index(df)
    m = df["KI67_in_hotspot"].fillna(False)
    if not m.any():
        return float("nan")
    sub = df.loc[m]
    n_pos = (sub["KI67_status"] == "Positive").sum()
    return round(100.0 * n_pos / len(sub), 2)
