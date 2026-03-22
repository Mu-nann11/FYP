from pathlib import Path
import re
import tifffile

CHANNELS = ["DAPI", "HER2", "PR", "ER"]


def crop_one_block(block_dir: Path, crop_root: Path, margin: int = 20, out_name_prefix: str = None):
    """Crop Cycle1 channels. Returns (y0, x0, y1, x1) window, or None on failure."""
    block_name = block_dir.name
    prefix = out_name_prefix or block_name

    paths = {}
    for ch in CHANNELS:
        pattern = f"{block_name}_{ch}*.tif"
        files = sorted(block_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        if not files:
            print(f"[{block_name}] 通道 {ch} 未找到匹配 {pattern}，跳过该块")
            return None
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

    y0, x0 = margin, margin
    y1, x1 = y0 + crop_h, x0 + crop_w

    print(f"[{block_name}] 原始尺寸: {sizes}")
    print(f"[{block_name}] 裁剪窗口: x={x0}:{x1}, y={y0}:{y1} -> {crop_w}×{crop_h}")

    for ch, arr in arrays.items():
        h, w = arr.shape[:2]
        if h < y1 or w < x1:
            print(f"[{block_name}] 通道 {ch} 尺寸过小 ({h}x{w})，无法按统一窗口裁剪，跳过该块")
            return None

    out_block_dir = crop_root / prefix
    out_block_dir.mkdir(parents=True, exist_ok=True)

    for ch, arr in arrays.items():
        cropped = arr[y0:y1, x0:x1]
        out_path = out_block_dir / f"{prefix}_{ch}_crop.tif"
        tifffile.imwrite(str(out_path), cropped)
        print(f"[{block_name}] 已保存: {out_path}")

    return (y0, x0, y1, x1)


def crop_cycle2(cycle2_dir: Path, crop_root: Path, block_base: str, window):
    """Crop Cycle2 composite using the same window as Cycle1."""
    y0, x0, y1, x1 = window

    # find composite tif
    candidates = list(cycle2_dir.glob("*_Composite.tif")) + list(cycle2_dir.glob("*.tif"))
    candidates = [f for f in candidates if f.is_file()]
    if not candidates:
        print(f"[{cycle2_dir.name}] 未找到 Cycle2 tif，跳过")
        return

    comp_path = candidates[0]
    arr = tifffile.imread(str(comp_path))
    h, w = arr.shape[-2], arr.shape[-1]

    if h < y1 or w < x1:
        print(f"[{cycle2_dir.name}] Cycle2 尺寸过小 ({h}x{w})，使用 resize 对齐")
        import cv2
        import numpy as np
        target_h, target_w = y1 - y0, x1 - x0
        if arr.ndim == 2:
            arr = cv2.resize(arr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            cropped = arr
        else:
            cropped = np.stack([
                cv2.resize(arr[i], (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                for i in range(arr.shape[0])
            ])
    else:
        cropped = arr[..., y0:y1, x0:x1] if arr.ndim > 2 else arr[y0:y1, x0:x1]

    out_dir = crop_root / block_base / "cycle2"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{block_base}_composite_crop.tif"
    tifffile.imwrite(str(out_path), cropped)
    print(f"[{cycle2_dir.name}] Cycle2 已保存: {out_path}")


def crop_all_blocks(config):
    stitched_root = Path(config["CROP_INPUT_DIR"])
    crop_root = Path(config["CROP_OUTPUT_DIR"])
    margin = int(config.get("CROP_MARGIN", 20))

    crop_root.mkdir(parents=True, exist_ok=True)
    block_pat = re.compile(r"^[A-Za-z0-9_-]+$")
    skip_names = {"logs", "log", "crop", "crop_result", "stitched_result", "stitched_results"}

    for entry in sorted(stitched_root.iterdir()):
        if not entry.is_dir() or not block_pat.match(entry.name):
            continue

        has_tif = any(entry.glob("*.tif"))
        if has_tif:
            # 旧结构：直接是 block_dir
            if entry.name.lower() not in skip_names:
                crop_one_block(entry, crop_root=crop_root, margin=margin)
        else:
            # 新结构：dataset 层
            dataset_name = entry.name
            dataset_crop_root = crop_root / dataset_name

            # 收集所有 Cycle1 目录，配对 Cycle2
            cycle1_dirs = {}
            cycle2_dirs = {}
            plain_dirs = []

            for block_dir in sorted(entry.iterdir()):
                if not block_dir.is_dir() or not block_pat.match(block_dir.name):
                    continue
                if block_dir.name.lower() in skip_names:
                    continue
                name = block_dir.name
                if "_Cycle1" in name or "_cycle1" in name:
                    base = re.sub(r"[_]?[Cc]ycle1$", "", name)
                    cycle1_dirs[base] = block_dir
                elif "_Cycle2" in name or "_cycle2" in name:
                    base = re.sub(r"[_]?[Cc]ycle2$", "", name)
                    cycle2_dirs[base] = block_dir
                else:
                    plain_dirs.append(block_dir)

            # 处理有 Cycle 的 block（TMAd）
            for base, c1_dir in cycle1_dirs.items():
                # Cycle1 输出到 cycle1/ 子目录，文件名前缀用 base
                c1_crop_root = dataset_crop_root / base / "cycle1"
                window = crop_one_block(c1_dir, crop_root=c1_crop_root, margin=margin, out_name_prefix=base)

                # Cycle2
                if base in cycle2_dirs and window is not None:
                    crop_cycle2(cycle2_dirs[base], dataset_crop_root, base, window)
                elif base not in cycle2_dirs:
                    print(f"[{base}] 未找到对应 Cycle2 目录，跳过 Cycle2")

            # 处理没有 Cycle 的 block（TMAe）
            for block_dir in plain_dirs:
                crop_one_block(block_dir, crop_root=dataset_crop_root, margin=margin)

