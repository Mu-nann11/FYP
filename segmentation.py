import torch
import numpy as np
import tifffile
from skimage.segmentation import expand_labels
from skimage.segmentation import watershed
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects
from skimage.measure import label
import cv2

from utils import normalize_to_uint16
from config import config

USE_GPU = config.get("SEGMENTATION.USE_GPU", torch.cuda.is_available())
_MODEL = None
_STARDIST_MODEL = None


def get_nuclei_model():
    global _MODEL
    if _MODEL is None:
        try:
            from cellpose import models as cp_models
        except ImportError:
            raise ImportError(
                "cellpose 未安装。请运行: pip install cellpose\n"
                "或在 environment.yml 中添加 cellpose 依赖。"
            )
        print("Init CellposeModel once. USE_GPU =", USE_GPU)
        _MODEL = cp_models.CellposeModel(gpu=USE_GPU, model_type="nuclei")
    return _MODEL


def segment_nuclei(dapi_img):
    """
    输入可以是 uint16。
    分割前转为 float32 供模型计算，但不改变磁盘上的 16-bit 文件策略。
    """
    model = get_nuclei_model()
    img = dapi_img.astype(np.float32)

    masks, flows, styles = model.eval(
        img,
        channels=config.get("SEGMENTATION.CHANNELS", [0, 0]),
        diameter=config.get("SEGMENTATION.DIAMETER", None),
        flow_threshold=config.get("SEGMENTATION.FLOW_THRESHOLD", 0.4),
        cellprob_threshold=config.get("SEGMENTATION.CELLPROB_THRESHOLD", 0.0),
    )
    return masks


def get_stardist_model():
    global _STARDIST_MODEL
    if _STARDIST_MODEL is None:
        from stardist.models import StarDist2D
        _STARDIST_MODEL = StarDist2D.from_pretrained("2D_versatile_fluo")
    return _STARDIST_MODEL


def segment_nuclei_stardist(dapi_img):
    from csbdeep.utils import normalize

    img = dapi_img.astype(np.float32)
    img = normalize(img, 1, 99.8, axis=None)

    model = get_stardist_model()
    labels, _ = model.predict_instances(img)
    return labels.astype(np.int32)


def segment_nuclei_watershed(
    dapi_img,
    min_area_px: int = 64,
    blur_ksize: int = 3,
):
    img = dapi_img.astype(np.float32)
    img -= float(img.min())
    vmax = float(img.max())
    if vmax > 0:
        img /= vmax

    if blur_ksize and blur_ksize > 1:
        if blur_ksize % 2 == 0:
            blur_ksize += 1
        img_blur = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)
    else:
        img_blur = img

    thr = float(threshold_otsu(img_blur))
    fg = img_blur > thr
    fg = remove_small_objects(fg, min_size=int(min_area_px))

    if not np.any(fg):
        return np.zeros(dapi_img.shape[:2], dtype=np.int32)

    dist = cv2.distanceTransform(fg.astype(np.uint8), cv2.DIST_L2, 3)
    peak = dist > (0.5 * float(dist.max()))
    markers = label(peak)
    if int(markers.max()) == 0:
        markers = label(fg)

    seg = watershed(-dist, markers=markers, mask=fg)
    seg = seg.astype(np.int32)
    return seg


def segment_nuclei_by_method(dapi_img, method: str):
    m = str(method).strip().lower()
    if m in ("cellpose", "cp", "nuclei"):
        return segment_nuclei(dapi_img)
    if m in ("stardist", "sd"):
        return segment_nuclei_stardist(dapi_img)
    if m in ("watershed", "ws"):
        return segment_nuclei_watershed(dapi_img)
    raise ValueError("unknown segmentation method: %s" % method)


def get_cytoplasm_masks(nuclei_masks, expansion_distance=None):
    """
    基于核掩膜进行扩张，估算胞质区域。
    expansion_distance: 扩张的像素距离。若为 None，则从配置读取。
    返回: (cell_masks, cyto_only_masks)
    - cell_masks: 包含核与质的整个细胞。
    - cyto_only_masks: 仅包含胞质（去掉了核的部分）。
    """
    if expansion_distance is None:
        expansion_distance = config.expansion_distance
    
    # 整个细胞的掩膜 (核 + 扩张出的质)
    cell_masks = expand_labels(nuclei_masks, distance=expansion_distance)
    
    # 仅胞质部分的掩膜 (cell_masks 减去 nuclei_masks)
    cyto_only_masks = cell_masks.copy()
    cyto_only_masks[nuclei_masks > 0] = 0
    
    return cell_masks, cyto_only_masks


def save_nuclei_overlay(dapi, masks, out_tif_path, cell_masks=None):
    """
    保存 16-bit TIFF overlay。
    如果提供了 cell_masks，还会绘制细胞边缘。
    """
    h, w = masks.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint16)

    rng = np.random.default_rng(42)
    max_lab = int(masks.max())
    
    # 绘制填色的核
    for lab in range(1, max_lab + 1):
        color = rng.integers(0, 65536, size=3, dtype=np.uint16)
        color_mask[masks == lab] = color

    if dapi.dtype != np.uint16:
        dapi16 = normalize_to_uint16(dapi)
    else:
        dapi16 = dapi

    dapi_rgb = np.stack([dapi16, dapi16, dapi16], axis=-1).astype(np.float32)
    overlay = 0.6 * dapi_rgb + 0.4 * color_mask.astype(np.float32)

    if cell_masks is not None:
        nuc = masks > 0
        cyto = (cell_masks > 0) & (~nuc)

        overlay[nuc] = 0.5 * overlay[nuc] + 0.5 * np.array([65535, 0, 0], dtype=np.float32)
        overlay[cyto] = 0.5 * overlay[cyto] + 0.5 * np.array([0, 65535, 0], dtype=np.float32)

        edges = np.zeros((h, w), dtype=np.uint8)
        for lab in range(1, int(cell_masks.max()) + 1):
            m = (cell_masks == lab).astype(np.uint8)
            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(edges, contours, -1, 255, 1)
        
        overlay[edges > 0] = [65535, 65535, 65535]

    overlay = overlay.clip(0, 65535).astype(np.uint16)
    tifffile.imwrite(str(out_tif_path), overlay)


def save_ki67_overlay(dapi, masks, ki67_img, out_tif_path):
    """
    Ki67 overlay：绿色=阳性细胞核，红色=阴性细胞核。
    阈值用 Otsu 自动确定。
    """
    from skimage.filters import threshold_otsu

    h, w = masks.shape
    if dapi.dtype != np.uint16:
        dapi16 = normalize_to_uint16(dapi)
    else:
        dapi16 = dapi

    overlay = np.stack([dapi16, dapi16, dapi16], axis=-1).astype(np.float32)

    # 计算 Ki67 阈值
    vals = ki67_img[masks > 0]
    thr = float(threshold_otsu(vals)) if vals.size > 1 else float(ki67_img.max() * 0.5)

    for lab in range(1, int(masks.max()) + 1):
        region = masks == lab
        mean_val = float(ki67_img[region].mean()) if region.any() else 0.0
        color = np.array([0, 65535, 0], dtype=np.float32) if mean_val > thr else np.array([65535, 0, 0], dtype=np.float32)
        overlay[region] = 0.5 * overlay[region] + 0.5 * color

    overlay = overlay.clip(0, 65535).astype(np.uint16)
    tifffile.imwrite(str(out_tif_path), overlay)


def save_ki67_hotspot_overlay(
    dapi: np.ndarray,
    masks: np.ndarray,
    ki67_img: np.ndarray,
    df,  # pd.DataFrame with KI67_status, KI67_in_hotspot, hotspot seeds
    seeds: list,
    radius_px: float,
    out_tif_path,
    per_hotspot_stats: list = None,
):
    """
    Ki67 hotspot overlay 可视化：
    - 灰底 DAPI
    - 绿色 = hotspot 内阳性细胞
    - 红色 = hotspot 内阴性细胞
    - 黄色半透明 = hotspot 外阳性细胞
    - 灰色 = hotspot 外阴性细胞
    - 青色圆环 = hotspot 区域边界
    - 左上角标注全局 + per-hotspot Ki67 指数
    """
    h, w = masks.shape

    if dapi.dtype != np.uint16:
        dapi16 = normalize_to_uint16(dapi)
    else:
        dapi16 = dapi

    # 基础灰底
    overlay = np.stack([dapi16, dapi16, dapi16], axis=-1).astype(np.float32)
    # 归一化到 [0,1] 方便混合
    overlay_01 = overlay / 65535.0

    # 准备颜色（归一化到 [0,1]）
    COLOR_POS_IN   = np.array([0.0, 1.0, 0.0])   # 绿：hotspot 内阳性
    COLOR_NEG_IN   = np.array([1.0, 0.0, 0.0])   # 红：hotspot 内阴性
    COLOR_POS_OUT  = np.array([1.0, 1.0, 0.0])   # 黄：hotspot 外阳性（弱显示）
    COLOR_NEG_OUT  = np.array([0.35, 0.35, 0.35]) # 灰：hotspot 外阴性
    ALPHA_IN  = 0.55
    ALPHA_OUT = 0.25

    # 构建 per-cell 查找
    has_status = "KI67_status" in df.columns
    has_hotspot = "KI67_in_hotspot" in df.columns

    for lab in range(1, int(masks.max()) + 1):
        region = masks == lab
        if not region.any():
            continue

        # 查找该 cell 的状态
        row = df[df["cell_id"] == lab]
        if row.empty:
            color = COLOR_NEG_OUT
            alpha = ALPHA_OUT
        else:
            is_pos = has_status and row.iloc[0].get("KI67_status") == "Positive"
            in_hs = has_hotspot and bool(row.iloc[0].get("KI67_in_hotspot", False))

            if in_hs:
                color = COLOR_POS_IN if is_pos else COLOR_NEG_IN
                alpha = ALPHA_IN
            else:
                color = COLOR_POS_OUT if is_pos else COLOR_NEG_OUT
                alpha = ALPHA_OUT

        overlay_01[region] = (1 - alpha) * overlay_01[region] + alpha * color

    # 画 hotspot 圆环
    overlay_u8 = (overlay_01 * 255).clip(0, 255).astype(np.uint8)

    for sx, sy in (seeds or []):
        cv2.circle(overlay_u8, (int(round(sx)), int(round(sy))),
                    int(round(radius_px)), (0, 255, 255), 2)  # 青色

    # 标注文字
    if has_status:
        n_pos_total = int((df["KI67_status"] == "Positive").sum())
        n_total = len(df)
        global_idx = round(100.0 * n_pos_total / max(n_total, 1), 1)

        if has_hotspot:
            hs_mask = df["KI67_in_hotspot"].fillna(False)
            hs_sub = df.loc[hs_mask]
            if not hs_sub.empty:
                hs_pos = int((hs_sub["KI67_status"] == "Positive").sum())
                hs_idx = round(100.0 * hs_pos / len(hs_sub), 1)
            else:
                hs_idx = float("nan")
        else:
            hs_idx = global_idx

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(overlay_u8, f"Global Ki67: {global_idx}%", (10, 30),
                    font, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(overlay_u8, f"Hotspot Ki67: {hs_idx}%", (10, 60),
                    font, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        # per-hotspot 标注
        if per_hotspot_stats:
            y_offset = 95
            for stat in per_hotspot_stats:
                label_text = (f"HS{stat['hotspot_id']}: {stat['ki67_index']}% "
                              f"({stat['n_positive']}/{stat['n_cells']})")
                cv2.putText(overlay_u8, label_text, (10, y_offset),
                            font, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
                y_offset += 25

    # 保存 TIFF + PNG
    tifffile.imwrite(str(out_tif_path), overlay_u8)
    png_path = str(out_tif_path).replace(".tif", ".png").replace(".tiff", ".png")
    cv2.imwrite(png_path, overlay_u8)


# =====================================================================
# SAM Mask Refinement
# =====================================================================

_SAM_MODEL = None


def _get_sam_model():
    """懒加载 SAM ViT-B 模型。需环境中有 segment-anything 包 + checkpoint。"""
    global _SAM_MODEL
    if _SAM_MODEL is not None:
        return _SAM_MODEL

    try:
        from segment_anything import sam_model_registry, SamPredictor
    except ImportError:
        raise ImportError(
            "segment-anything 未安装。请在 environment.yml 或 Dockerfile 中添加：\n"
            "  pip install git+https://github.com/facebookresearch/segment-anything.git"
        )

    sam_checkpoint = config.get("SEGMENTATION.SAM_CHECKPOINT", "/models/sam_vit_b_01ec64.pth")
    sam_model_type = config.get("SEGMENTATION.SAM_MODEL_TYPE", "vit_b")
    use_gpu = config.get("SEGMENTATION.USE_GPU", torch.cuda.is_available())

    import os
    if not os.path.isfile(sam_checkpoint):
        raise FileNotFoundError(
            f"SAM checkpoint 不存在: {sam_checkpoint}\n"
            f"请下载 ViT-B 权重并设置 SEGMENTATION.SAM_CHECKPOINT"
        )

    sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
    sam.to(device="cuda" if use_gpu else "cpu")
    _SAM_MODEL = SamPredictor(sam)
    print(f"SAM model loaded: {sam_model_type}, device={'cuda' if use_gpu else 'cpu'}")
    return _SAM_MODEL


def _dapi_to_rgb_uint8(dapi: np.ndarray) -> np.ndarray:
    """将单通道 DAPI 转为 3 通道 uint8 供 SAM 使用。"""
    if dapi.dtype == np.uint16:
        img = (dapi.astype(np.float32) / 65535.0 * 255.0).clip(0, 255).astype(np.uint8)
    elif dapi.dtype == np.float64 or dapi.dtype == np.float32:
        vmin, vmax = float(dapi.min()), float(dapi.max())
        if vmax > vmin:
            img = ((dapi - vmin) / (vmax - vmin) * 255.0).clip(0, 255).astype(np.uint8)
        else:
            img = np.zeros_like(dapi, dtype=np.uint8)
    else:
        img = dapi.astype(np.uint8)
    return np.stack([img, img, img], axis=-1)


def refine_masks_with_sam(
    dapi: np.ndarray,
    initial_masks: np.ndarray,
    min_area: int = 50,
    box_pad: int = 3,
    max_masks: int = 0,
) -> np.ndarray:
    """
    用 SAM 对 cellpose/StarDist/watershed 的初始 mask 进行边界精炼。

    流程：
    1. 对每个初始 mask 提取 bounding box
    2. 用 bbox 作为 SAM prompt，获取精炼 mask
    3. SAM mask 与原始 label 合并，重叠区域取最高 IoU 对应的 label
    4. 未被 SAM 覆盖的区域保留原始 mask

    参数：
        dapi: DAPI 图像
        initial_masks: 初始分割 label mask
        min_area: 忽略面积小于此的初始 mask
        box_pad: bbox 扩展像素（避免边界裁切）
        max_masks: 最大处理数量，0=不限

    返回：
        refined_masks: 精炼后的 label mask（int32）
    """
    predictor = _get_sam_model()

    h, w = initial_masks.shape
    rgb = _dapi_to_rgb_uint8(dapi)
    predictor.set_image(rgb)

    from skimage.measure import regionprops

    props = regionprops(initial_masks)
    n_total = len(props)
    if n_total == 0:
        return initial_masks.copy()

    # 过滤小区域 + 排序（大区域优先）
    props = [p for p in props if p.area >= min_area]
    props.sort(key=lambda p: p.area, reverse=True)

    if max_masks > 0:
        props = props[:max_masks]

    print(f"SAM refinement: {len(props)}/{n_total} masks to refine (min_area={min_area})")

    # 精炼结果：confidence map + label map
    refined = np.zeros((h, w), dtype=np.int32)
    confidence = np.zeros((h, w), dtype=np.float32)

    for p in props:
        label_id = p.label
        minr, minc, maxr, maxc = p.bbox

        # 扩展 bbox
        box = np.array([
            max(0, minc - box_pad),
            max(0, minr - box_pad),
            min(w - 1, maxc + box_pad),
            min(h - 1, maxr + box_pad),
        ])

        masks_sam, scores_sam, _ = predictor.predict(
            box=box,
            multimask_output=True,
        )

        if len(masks_sam) == 0:
            # SAM 没有输出，保留原始
            region = initial_masks == label_id
            refined[region] = label_id
            confidence[region] = 1.0
            continue

        # 取最高分 mask
        best_idx = np.argmax(scores_sam)
        sam_mask = masks_sam[best_idx]
        sam_score = float(scores_sam[best_idx])

        # 只在原始 mask 区域内更新（SAM 可能扩出额外区域）
        orig_region = initial_masks == label_id
        update_region = sam_mask & orig_region

        if update_region.sum() > 0:
            # 高置信区域用 SAM 结果
            higher_conf = sam_score > confidence[update_region]
            if higher_conf.any():
                refined[update_region & (sam_score > confidence)] = label_id
                confidence[update_region & (sam_score > confidence)] = sam_score

        # 原始区域中未被 SAM 覆盖的部分保留
        fallback = orig_region & (refined == 0)
        refined[fallback] = label_id
        confidence[fallback] = 1.0

    # 确保 label 连续
    unique_labels = np.unique(refined)
    unique_labels = unique_labels[unique_labels != 0]
    remap = {old: new for new, old in enumerate(unique_labels, 1)}
    remap[0] = 0
    refined = np.vectorize(remap.get)(refined).astype(np.int32)

    print(f"SAM refinement done: {len(unique_labels)} labels in output")
    return refined
