from pathlib import Path
import tifffile
import cv2
import numpy as np

from utils import normalize_to_uint16

CHANNELS = ["DAPI", "HER2", "PR", "ER"]
# 容器内裁剪结果目录，对应宿主机 ./results/crop
CROP_ROOT = Path("/results/crop")


def preprocess_16bit(img):
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

    if img.dtype != np.uint16:
        img = img.astype(np.uint16)

    return img


def load_block(block_name: str, do_preprocess: bool = True):
    """
    返回 dict: {channel: image_array}
    读取路径:
      /results/crop/<block_name>/<block_name>_<channel>_crop.tif

    默认读取后保持 16-bit。
    """
    block_dir = CROP_ROOT / block_name
    if not block_dir.exists():
        raise FileNotFoundError(f"裁剪结果目录不存在: {block_dir}")

    imgs = {}
    for ch in CHANNELS:
        path = block_dir / f"{block_name}_{ch}_crop.tif"
        if not path.exists():
            raise FileNotFoundError(f"缺少通道文件: {path}")

        img = tifffile.imread(str(path))

        if do_preprocess:
            img = preprocess_16bit(img)
        else:
            if img.dtype != np.uint16:
                img = normalize_to_uint16(img)

        imgs[ch] = img

    return imgs


if __name__ == "__main__":
    block_name = "G6"
    imgs = load_block(block_name, do_preprocess=True)
    print(f"Loaded block {block_name}:")
    for ch, im in imgs.items():
        print(
            f" {ch}: shape={im.shape}, dtype={im.dtype}, "
            f"min={im.min()}, max={im.max()}"
        )
