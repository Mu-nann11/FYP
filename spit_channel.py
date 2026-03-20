import tifffile
import numpy as np
from skimage import filters, measure, morphology
import argparse
import os
from pathlib import Path

def analyze_channel(img, min_size=50):
    """输入单通道图像，返回阳性区域面积和估计的核数量"""
    # 阈值分割（Otsu）
    thresh = filters.threshold_otsu(img)
    binary = img > thresh
    # 移除小杂点
    cleaned = morphology.remove_small_objects(binary, min_size=min_size)
    # 标记连通域
    labeled = measure.label(cleaned)
    props = measure.regionprops(labeled)
    nuclei_count = len(props)
    area = cleaned.sum()
    return area, nuclei_count

def normalize_path(path: str) -> str:
    path = (path or "").strip().strip("\"").strip("'")
    if not path:
        raise ValueError("path is required")

    path = path.replace("\\", "/")

    if ":/" in path:
        marker = "/Code/"
        idx = path.lower().find(marker.lower())
        if idx != -1:
            path = "/app/" + path[idx + len(marker):]
        else:
            raise ValueError(
                "Windows 路径无法在容器内直接访问。请传入容器内路径，例如 /app/data/... 或相对路径 data/..."
            )

    if path.startswith("/app/"):
        return path
    if path.startswith("/"):
        return path
    return f"/app/{path}"

def _extract_two_channels(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if img.ndim != 3:
        raise ValueError(f"Expected 3D image with 2 channels, got shape={img.shape}")

    if img.shape[0] == 2:
        return img[0], img[1]

    if img.shape[-1] == 2:
        return img[:, :, 0], img[:, :, 1]

    raise ValueError(f"Expected 2 channels in first or last axis, got shape={img.shape}")


def split_dir(input_dir: str, out_dir: str, ch0_name: str, ch1_name: str, overwrite: bool) -> None:
    in_dir = Path(normalize_path(input_dir))
    out_root = Path(normalize_path(out_dir))

    if not in_dir.exists() or not in_dir.is_dir():
        raise FileNotFoundError(f"Input dir not found: {in_dir}")

    ch0 = str(ch0_name).strip() or "CH0"
    ch1 = str(ch1_name).strip() or "CH1"

    out0 = out_root / ch0
    out1 = out_root / ch1
    out0.mkdir(parents=True, exist_ok=True)
    out1.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() in (".tif", ".tiff")])
    if not files:
        raise FileNotFoundError(f"No tif files found in: {in_dir}")

    for p in files:
        if p.name.startswith("TileConfiguration"):
            continue

        img = tifffile.imread(str(p))
        c0, c1 = _extract_two_channels(img)

        out0_path = out0 / f"{p.stem}_{ch0}.tif"
        out1_path = out1 / f"{p.stem}_{ch1}.tif"

        if not overwrite and (out0_path.exists() or out1_path.exists()):
            continue

        tifffile.imwrite(str(out0_path), c0)
        tifffile.imwrite(str(out1_path), c1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path")
    parser.add_argument("--input-dir")
    parser.add_argument("--out-dir")
    parser.add_argument("--ch0-name", default="DAPI")
    parser.add_argument("--ch1-name", default="Ki67")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--min-size", type=int, default=50)
    args = parser.parse_args()

    if args.input_dir and args.out_dir:
        split_dir(
            input_dir=args.input_dir,
            out_dir=args.out_dir,
            ch0_name=args.ch0_name,
            ch1_name=args.ch1_name,
            overwrite=bool(args.overwrite),
        )
        print("✅ 通道拆分完成")
        return

    if not args.path:
        raise ValueError("Either --path or (--input-dir and --out-dir) is required")

    path = normalize_path(args.path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found in container: {path}")

    img = tifffile.imread(path)

    print(f"图像形状: {img.shape}")

    if img.ndim == 3:
        if img.shape[0] < img.shape[-1]:
            for i in range(img.shape[0]):
                ch = img[i]
                area, nuc = analyze_channel(ch, min_size=args.min_size)
                print(f"通道 {i}: 阳性面积={area}, 估计核数={nuc}")
        else:
            for i in range(img.shape[-1]):
                ch = img[:, :, i]
                area, nuc = analyze_channel(ch, min_size=args.min_size)
                print(f"通道 {i}: 阳性面积={area}, 估计核数={nuc}")
    else:
        print("图像不是多通道？请检查。")


if __name__ == "__main__":
    main()
