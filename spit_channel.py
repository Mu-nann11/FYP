import os
import tifffile as tiff
import numpy as np

def split_channels(img):
    """
    支持:
    (2, H, W) 或 (H, W, 2)
    """
    if img.ndim != 3:
        raise ValueError(f"不是多通道图: shape={img.shape}")

    if img.shape[0] == 2:
        ch1, ch2 = img[0], img[1]
    elif img.shape[-1] == 2:
        ch1, ch2 = img[..., 0], img[..., 1]
    else:
        raise ValueError(f"不是2通道图: shape={img.shape}")

    return ch1, ch2


def process_folder(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    dapi_dir = os.path.join(output_dir, "DAPI")
    ki67_dir = os.path.join(output_dir, "KI67")

    os.makedirs(dapi_dir, exist_ok=True)
    os.makedirs(ki67_dir, exist_ok=True)

    for root, _, filenames in os.walk(input_dir):
        files = [f for f in filenames if f.endswith(".tif")]
        for f in files:
            path = os.path.join(root, f)
            print(f"处理: {path}")

            img = tiff.imread(path)
            print(f"shape: {img.shape}")

            try:
                ch1, ch2 = split_channels(img)
            except Exception as e:
                print(f"跳过 {f}: {e}")
                continue

            base = os.path.splitext(f)[0]

            dapi_path = os.path.join(dapi_dir, base + "_DAPI.tif")
            ki67_path = os.path.join(ki67_dir, base + "_KI67.tif")

            tiff.imwrite(dapi_path, ch1)
            tiff.imwrite(ki67_path, ch2)

    print("完成！")


if __name__ == "__main__":
    input_dir = "/data/Raw_Data/TMAd/Cycle2"
    output_dir = "/results/cycle2_split"

    process_folder(input_dir, output_dir)