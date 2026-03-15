from pathlib import Path
import re
import tifffile

CHANNELS = ["DAPI", "HER2", "PR", "ER"]


def crop_one_block(block_dir: Path, crop_root: Path, margin: int = 20):
    block_name = block_dir.name

    paths = {}
    for ch in CHANNELS:
        pattern = f"{block_name}_{ch}*.tif"
        files = sorted(block_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        if not files:
            print(f"[{block_name}] 通道 {ch} 未找到匹配 {pattern}，跳过该块")
            return
        paths[ch] = files[0]

    sizes = {}
    arrays = {}
    for ch, p in paths.items():
        arr = tifffile.imread(str(p))
        arrays[ch] = arr
        h, w = arr.shape[:2]
        sizes[ch] = (h, w, str(arr.dtype))

    min_h = min(v[0] for v in sizes.values())
    min_w = min(v[1] for v in sizes.values())

    crop_h = max(min_h - 2 * margin, 1)
    crop_w = max(min_w - 2 * margin, 1)

    y0 = margin
    x0 = margin
    y1 = y0 + crop_h
    x1 = x0 + crop_w

    print(f"[{block_name}] 原始尺寸: {sizes}")
    print(f"[{block_name}] 裁剪窗口: x={x0}:{x1}, y={y0}:{y1} -> {crop_w}×{crop_h}")

    for ch, arr in arrays.items():
        h, w = arr.shape[:2]
        if h < y1 or w < x1:
            print(f"[{block_name}] 通道 {ch} 尺寸过小 ({h}x{w})，无法按统一窗口裁剪，跳过该块")
            return

    out_block_dir = crop_root / block_name
    out_block_dir.mkdir(parents=True, exist_ok=True)

    for ch, arr in arrays.items():
        cropped = arr[y0:y1, x0:x1]
        out_name = f"{block_name}_{ch}_crop.tif"
        out_path = out_block_dir / out_name
        tifffile.imwrite(str(out_path), cropped)
        print(f"[{block_name}] 已保存: {out_path}")


def crop_all_blocks(config):
    stitched_root = Path(config["CROP_INPUT_DIR"])
    crop_root = Path(config["CROP_OUTPUT_DIR"])
    margin = int(config.get("CROP_MARGIN", 20))

    crop_root.mkdir(parents=True, exist_ok=True)
    block_pat = re.compile(r"^[A-Za-z0-9_-]+$")

    for block_dir in sorted(stitched_root.iterdir()):
        if not block_dir.is_dir():
            continue

        name = block_dir.name
        if name.lower() in ("logs", "log", "crop", "crop_result", "stitched_result", "stitched_results"):
            continue

        if not block_pat.match(name):
            continue

        crop_one_block(block_dir, crop_root=crop_root, margin=margin)
