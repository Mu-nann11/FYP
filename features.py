from pathlib import Path
import numpy as np
import pandas as pd
from skimage.measure import regionprops, regionprops_table
import cv2
from utils import q90
from config import config

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

    return df

def score_markers(df):
    """
    使用配置文件中的阈值进行自动评分。
    """
    if df.empty:
        return df

    # 从配置读取阈值
    pos_thr = config.get("SCORING.POSITIVE_THRESHOLD", 5000)
    her2_thrs = config.get("SCORING.HER2_THRESHOLDS", {"3+": 15000, "2+": 8000, "1+": 3000})
    
    # ER/PR 评分 (核内均值)
    for marker in ["ER", "PR"]:
        col = "%s_nuc_mean" % marker
        if col in df.columns:
            df["%s_status" % marker] = df[col].apply(lambda x: "Positive" if x > pos_thr else "Negative")
    
    # HER2 评分 (胞质内均值)
    if "HER2_cyto_mean" in df.columns:
        def score_her2(val):
            if val > her2_thrs.get("3+", 15000): return "3+"
            if val > her2_thrs.get("2+", 8000): return "2+"
            if val > her2_thrs.get("1+", 3000): return "1+"
            return "0"
        df["HER2_score"] = df["HER2_cyto_mean"].apply(score_her2)
        
    return df
