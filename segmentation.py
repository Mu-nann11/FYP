import torch
import numpy as np
import tifffile
from cellpose import models
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
        print("Init CellposeModel once. USE_GPU =", USE_GPU)
        _MODEL = models.CellposeModel(gpu=USE_GPU, model_type="nuclei")
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
