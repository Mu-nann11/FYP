from pathlib import Path
import tifffile
import cv2
import numpy as np
from typing import Dict, Any

from utils import normalize_to_uint16

# 容器内裁剪结果目录，对应宿主机 ./results/crop
CROP_ROOT = Path("/results/crop")

DATASETS = {
    "TMAe": {
        "mode": "single_cycle",
        "cycle1_channels": ["DAPI", "HER2", "PR", "ER"],
    },
    "TMAd": {
        "mode": "multi_cycle",
        "cycle1_channels": ["DAPI", "HER2", "PR", "ER"],
        "cycle2_channels": ["DAPI", "KI67"],
        # 如果 cycle2 是合成复合通道，可设置 True
        "cycle2_composite": True,
    },
}


def preprocess_16bit(img: np.ndarray) -> np.ndarray:
    """
    保持 16-bit 的预处理：
    1) 归一化到 uint16
    2) 16-bit CLAHE
    3) 轻微高斯模糊
    """
    if img.dtype != np.uint16:
        img = normalize_to_uint16(img)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    img = cv2.GaussianBlur(img, (3, 3), 0)

    return img.astype(np.uint16)


def _read_one_image(path: Path, do_preprocess: bool = True) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"缺少通道文件: {path}")

    img = tifffile.imread(str(path))
    if do_preprocess:
        return preprocess_16bit(img)

    return normalize_to_uint16(img) if img.dtype != np.uint16 else img


def _load_cycle(
    cycle_dir: Path,
    block_name: str,
    channels,
    composite: bool = False,
    do_preprocess: bool = True,
) -> Dict[str, np.ndarray]:
    imgs: Dict[str, np.ndarray] = {}
    # 兼容 cycle 目录下有/没有 block 子目录两种结构
    block_subdir = cycle_dir / block_name
    if block_subdir.is_dir():
        cycle_dir = block_subdir
    if composite:
        comp_path = cycle_dir / f"{block_name}_composite_crop.tif"
        if not comp_path.exists():
            # 尝试备选模式
            comp_path = cycle_dir / f"{block_name}_cycle2_crop.tif"
        if not comp_path.exists():
            raise FileNotFoundError(f"缺少 cycle2 composite 文件: {comp_path}")

        comp = tifffile.imread(str(comp_path))

        # 兼容多种存储形状
        if comp.ndim == 3 and comp.shape[0] == len(channels):
            for idx, ch in enumerate(channels):
                channel_img = comp[idx]
                imgs[ch] = preprocess_16bit(channel_img) if do_preprocess else normalize_to_uint16(channel_img)
        elif comp.ndim == 3 and comp.shape[-1] == len(channels):
            for idx, ch in enumerate(channels):
                channel_img = comp[..., idx]
                imgs[ch] = preprocess_16bit(channel_img) if do_preprocess else normalize_to_uint16(channel_img)
        elif len(channels) >= 2 and comp.ndim == 2:
            # 单通道 composite → 同时赋给所有 channel（如 DAPI/KI67 共用）
            proc = preprocess_16bit(comp) if do_preprocess else normalize_to_uint16(comp)
            for ch in channels:
                imgs[ch] = proc
        else:
            raise ValueError("cycle2 composite 图像格式不符合预期，需要 2 通道多层 tiff")

    else:
        for ch in channels:
            path = cycle_dir / f"{block_name}_{ch}_crop.tif"
            imgs[ch] = _read_one_image(path, do_preprocess=do_preprocess)

    return imgs



def load_block(dataset_name: str, block_name: str, do_preprocess: bool = True) -> Dict[str, Any]:
    if dataset_name not in DATASETS:
        raise ValueError(f"未知数据集: {dataset_name}. 可选: {list(DATASETS.keys())}")

    spec = DATASETS[dataset_name]
    mode = spec["mode"]

    block_dir = CROP_ROOT / dataset_name / block_name
    if not block_dir.exists():
        raise FileNotFoundError(f"block 目录不存在: {block_dir}")

    result: Dict[str, Any] = {
        "dataset": dataset_name,
        "block": block_name,
        "mode": mode,
    }

    if mode == "single_cycle":
        result["cycle1"] = _load_cycle(
            block_dir,
            block_name,
            spec["cycle1_channels"],
            do_preprocess=do_preprocess,
        )

    else:
        cycle1_dir = block_dir / "cycle1"
        cycle2_dir = block_dir / "cycle2"
        if not cycle1_dir.exists():
            raise FileNotFoundError(f"缺少 cycle1 目录: {cycle1_dir}")
        if not cycle2_dir.exists():
            raise FileNotFoundError(f"缺少 cycle2 目录: {cycle2_dir}")

        result["cycle1"] = _load_cycle(
            cycle1_dir,
            block_name,
            spec["cycle1_channels"],
            do_preprocess=do_preprocess,
        )

        result["cycle2"] = _load_cycle(
            cycle2_dir,
            block_name,
            spec["cycle2_channels"],
            composite=spec.get("cycle2_composite", False),
            do_preprocess=do_preprocess,
        )

    return result


def verify_block(dataset_name: str, block_name: str) -> bool:
    """轻度验证文件存在性，返回 True 表示通过"""
    try:
        data = load_block(dataset_name, block_name, do_preprocess=False)
    except Exception as e:
        print(f"验证失败: {dataset_name}/{block_name} -> {e}")
        return False

    for c in ["cycle1", "cycle2"]:
        if c in data:
            for ch, img in data[c].items():
                if not isinstance(img, np.ndarray):
                    print(f"验证失败: {dataset_name}/{block_name}/{c}/{ch} 不是 ndarray")
                    return False
    return True


if __name__ == "__main__":
    for dataset_name, block_name in [("TMAe", "A1"), ("TMAd", "A1")]:
        try:
            data = load_block(dataset_name, block_name, do_preprocess=True)
            print(f"\nLoaded {dataset_name}/{block_name}")
            for cycle_name in ["cycle1", "cycle2"]:
                if cycle_name in data:
                    print(f"  {cycle_name}:")
                    for ch, im in data[cycle_name].items():
                        print(
                            f"    {ch}: shape={im.shape}, dtype={im.dtype}, "
                            f"min={im.min()}, max={im.max()}"
                        )
        except Exception as e:
            print(f"Failed: {dataset_name}/{block_name}: {e}")

