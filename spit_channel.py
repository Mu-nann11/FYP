"""
Cycle2 双通道 Composite → 按 block 拆分为 DAPI / KI67。

目录结构期望：
  <cycle2>/
    <block>/          ← 如 A3, B5 ...
      *.tif           ← 根目录下的双通道 Composite
      DAPI/           ← 拆分输出
      KI67/           ← 拆分输出
      Composite_source/ ← 归档源文件（默认）

默认认为：第 0 通道 → Ki67，第 1 通道 → DAPI（写入时交换）。
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import tifffile as tiff

COMPOSITE_SOURCE_DIR = "Composite_source"

# 跳过的子目录名（这些不是 block）
_SKIP_DIRS = {"DAPI", "KI67", COMPOSITE_SOURCE_DIR}


def split_channels(img):
    """
    支持 (2, H, W) 或 (H, W, 2)
    返回 (ch1, ch2)
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


def split_one_block(block_dir: Path, dry_run: bool = False,
                    remove_sources: bool = False,
                    swap_dapi_ki67: bool = True):
    """
    拆分一个 block 目录下的所有 *.tif / *.tiff。
    
    默认交换 DAPI/Ki67：
      原始 ch0 → 写入 KI67/
      原始 ch1 → 写入 DAPI/
    --no-swap-dapi-ki67 时：
      ch0 → DAPI/
      ch1 → KI67/
    """
    dapi_out = block_dir / "DAPI"
    ki67_out = block_dir / "KI67"

    tif_files = sorted(
        f for f in block_dir.iterdir()
        if f.suffix.lower() in (".tif", ".tiff")
        and f.is_file()
        # 跳过已经拆分过的文件（文件名含 _dapi / _ki67）
        and "_dapi" not in f.stem.lower()
        and "_ki67" not in f.stem.lower()
    )

    if not tif_files:
        return 0

    if not dry_run:
        dapi_out.mkdir(parents=True, exist_ok=True)
        ki67_out.mkdir(parents=True, exist_ok=True)

    count = 0
    for fpath in tif_files:
        try:
            img = tiff.imread(str(fpath))
            ch0, ch1 = split_channels(img)
        except Exception as e:
            print(f"  ⚠️ 跳过 {fpath.name}: {e}")
            continue

        base = fpath.stem

        if swap_dapi_ki67:
            # 默认：ch0→Ki67, ch1→DAPI（修正通道反了的情况）
            dapi_data, ki67_data = ch1, ch0
        else:
            # 不交换：ch0→DAPI, ch1→Ki67
            dapi_data, ki67_data = ch0, ch1

        dapi_path = dapi_out / f"{base}_dapi.tif"
        ki67_path = ki67_out / f"{base}_ki67.tif"

        if dry_run:
            print(f"  [dry-run] {fpath.name} → DAPI/{dapi_path.name} + KI67/{ki67_path.name}")
        else:
            tiff.imwrite(str(dapi_path), dapi_data)
            tiff.imwrite(str(ki67_path), ki67_data)
            print(f"  ✓ {fpath.name} → DAPI/ + KI67/")

            # 归档或删除源文件
            if remove_sources:
                fpath.unlink()
                print(f"    🗑️ 删除源文件: {fpath.name}")
            else:
                archive_dir = block_dir / COMPOSITE_SOURCE_DIR
                archive_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(fpath), str(archive_dir / fpath.name))
                print(f"    📦 归档到 {COMPOSITE_SOURCE_DIR}/: {fpath.name}")

        count += 1

    return count


def run_cycle2_split(cycle_dir: Path, dry_run: bool = False,
                     remove_sources: bool = False,
                     swap_dapi_ki67: bool = True):
    """
    遍历 Cycle2 下的子目录，对每个 block 执行拆分。
    跳过 DAPI / KI67 / Composite_source。
    """
    if not cycle_dir.is_dir():
        print(f"❌ Cycle 目录不存在: {cycle_dir}")
        return

    print(f"📂 Cycle2 目录: {cycle_dir}")
    if dry_run:
        print("🔸 模式: dry-run（仅打印，不写入）")
    print()

    blocks = sorted(
        d for d in cycle_dir.iterdir()
        if d.is_dir() and d.name not in _SKIP_DIRS
    )

    if not blocks:
        print("⚠️ 未找到任何 block 子目录")
        return

    total = 0
    for block_dir in blocks:
        n = split_one_block(
            block_dir,
            dry_run=dry_run,
            remove_sources=remove_sources,
            swap_dapi_ki67=swap_dapi_ki67,
        )
        if n:
            print(f"  📁 {block_dir.name}: {n} 个文件已处理")
            total += n
        else:
            print(f"  ⏭️ {block_dir.name}: 无待处理文件（已跳过）")

    print(f"\n✅ 完成！共处理 {total} 个文件，覆盖 {len(blocks)} 个 block。")


def main():
    parser = argparse.ArgumentParser(
        description="Cycle2 Composite → DAPI / KI67 通道拆分"
    )

    # 方式一：直接指定 cycle 目录
    parser.add_argument("--cycle-dir", default=None,
                        help="直接指定 Cycle2 的完整路径")

    # 方式二：通过 root + subdir / dataset + cycle 拼出路径
    parser.add_argument("--root", default=None,
                        help="Raw_Data 根目录（默认 /data/Raw_Data）")
    parser.add_argument("--raw-subdir", default=None,
                        help="Raw_Data 下的子目录，如 'TMAd/Cycle2'")
    parser.add_argument("--dataset", default=None, choices=["TMAe", "TMAd"],
                        help="数据集名称，配合 --cycle 使用")
    parser.add_argument("--cycle", default=None,
                        help="Cycle 名称，如 Cycle2，配合 --dataset 使用")

    parser.add_argument("--dry-run", action="store_true",
                        help="仅打印将要执行的操作，不实际写入")
    parser.add_argument("--remove-sources", action="store_true",
                        help="拆分后直接删除源文件，不归档到 Composite_source/")
    parser.add_argument("--no-swap-dapi-ki67", action="store_true",
                        help="不交换 DAPI/Ki67 通道（默认 ch0→Ki67, ch1→DAPI；启用此选项后 ch0→DAPI, ch1→Ki67）")

    args = parser.parse_args()

    # 解析 cycle_dir
    if args.cycle_dir:
        cycle_dir = Path(args.cycle_dir).expanduser().resolve()
    elif args.raw_subdir:
        default_root = args.root or "/data/Raw_Data"
        cycle_dir = Path(default_root) / args.raw_subdir
    elif args.dataset and args.cycle:
        default_root = args.root or "/data/Raw_Data"
        cycle_dir = Path(default_root) / args.dataset / args.cycle
    else:
        # 默认路径
        cycle_dir = Path("/data/Raw_Data/TMAd/Cycle2")

    run_cycle2_split(
        cycle_dir=cycle_dir,
        dry_run=args.dry_run,
        remove_sources=args.remove_sources,
        swap_dapi_ki67=not args.no_swap_dapi_ki67,
    )


if __name__ == "__main__":
    main()
