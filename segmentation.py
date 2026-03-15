import torch
import numpy as np
import tifffile
from cellpose import models

USE_GPU = torch.cuda.is_available()
_MODEL = None


def get_nuclei_model():
    global _MODEL
    if _MODEL is None:
        print("Init CellposeModel once. USE_GPU =", USE_GPU)
        _MODEL = models.CellposeModel(gpu=USE_GPU, model_type="nuclei")
    return _MODEL


def normalize_to_uint16(img):
    img = img.astype(np.float32)
    img -= img.min()
    if img.max() > 0:
        img /= img.max()
    img = (img * 65535.0).clip(0, 65535)
    return img.astype(np.uint16)


def segment_nuclei(dapi_img):
    """
    输入可以是 uint16。
    分割前转为 float32 供模型计算，但不改变磁盘上的 16-bit 文件策略。
    """
    model = get_nuclei_model()
    img = dapi_img.astype(np.float32)

    masks, flows, styles = model.eval(
        img,
        channels=[0, 0],
        diameter=None,
        flow_threshold=0.4,
        cellprob_threshold=0.0,
    )
    return masks


def save_nuclei_overlay(dapi, masks, out_tif_path):
    """
    保存 16-bit TIFF overlay，而不是 8-bit PNG。
    """
    h, w = masks.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint16)

    rng = np.random.default_rng(42)
    for lab in range(1, int(masks.max()) + 1):
        color = rng.integers(0, 65536, size=3, dtype=np.uint16)
        color_mask[masks == lab] = color

    if dapi.dtype != np.uint16:
        dapi16 = normalize_to_uint16(dapi)
    else:
        dapi16 = dapi

    dapi_rgb = np.stack([dapi16, dapi16, dapi16], axis=-1)

    overlay = (
        0.5 * dapi_rgb.astype(np.float32) +
        0.5 * color_mask.astype(np.float32)
    ).clip(0, 65535).astype(np.uint16)

    tifffile.imwrite(str(out_tif_path), overlay)
