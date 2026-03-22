"""
将 cycle2_split 的扁平 DAPI/KI67 输出，按文件名前缀（如 A3_...）
分发到 Raw_Data/.../Cycle2/<block>/DAPI|KI67/ 目录。

用法示例：
  python organize_cycle2_split.py \
      --dapi-dir /results/cycle2_split/DAPI \
      --ki67-dir /results/cycle2_split/KI67 \
      --dest /data/Raw_Data/TMAd/Cycle2 \
      --move --dry-run
"""

import argparse
import os
import shutil
from pathlib import Path


def _block_prefix(filename: str) -> str:
    """
    从文件名提取 block 前缀。
    例如 'A3_001_s0_dapi.tif' → 'A3'
    """
    parts = filename.split("_")
    if len(parts) >= 1:
        return parts[0]
    return filename


def organize_channel(flat_dir: Path, channel_name: str,
                     dest_root: Path, do_move: bool, dry_run: bool):
    """
    将 flat_dir 下的文件按前缀分发到 dest_root/<block>/<channel_name>/
    """
    if not flat_dir.is_dir():
        print(f"⚠️ 目录不存在，跳过: {flat_dir}")
        return 0

    files = sorted(
        f for f in flat_dir.iterdir()
        if f.is_file() and f.suffix.lower() in (".tif", ".tiff")
    )

    if not files:
        print(f"  {channel_name}: 无文件")
        return 0

    count = 0
    for fpath in files:
        block = _block_prefix(fpath.stem)
        target_dir = dest_root / block / channel_name
        target_path = target_dir / fpath.name

        if target_path.exists():
            print(f"  ⚠️ 已存在，跳过: {target_path.relative_to(dest_root)}")
            continue

        if dry_run:
            action = "move" if do_move else "copy"
            print(f"  [dry-run] {action} {fpath.name} → {block}/{channel_name}/")
        else:
            target_dir.mkdir(parents=True, exist_ok=True)
            if do_move:
                shutil.move(str(fpath), str(target_path))
            else:
                shutil.copy2(str(fpath), str(target_path))
            print(f"  ✓ {fpath.name} → {block}/{channel_name}/")

        count += 1

    return count


def main():
    parser = argparse.ArgumentParser(
        description="将扁平的 DAPI/KI67 文件按 block 前缀归入目录结构"
    )
    parser.add_argument("--dapi-dir", required=True,
                        help="扁平 DAPI 文件所在目录")
    parser.add_argument("--ki67-dir", required=True,
                        help="扁平 KI67 文件所在目录")
    parser.add_argument("--dest", required=True,
                        help="目标根目录，如 /data/Raw_Data/TMAd/Cycle2")
    parser.add_argument("--move", action="store_true",
                        help="移动文件而非复制（默认复制）")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅打印，不执行")
    args = parser.parse_args()

    dest = Path(args.dest).expanduser().resolve()

    print(f"📂 目标目录: {dest}")
    if args.dry_run:
        print("🔸 模式: dry-run\n")

    n_dapi = organize_channel(
        Path(args.dapi_dir), "DAPI", dest, do_move=args.move, dry_run=args.dry_run
    )
    n_ki67 = organize_channel(
        Path(args.ki67_dir), "KI67", dest, do_move=args.move, dry_run=args.dry_run
    )

    print(f"\n✅ 完成！DAPI: {n_dapi} 个文件, KI67: {n_ki67} 个文件。")


if __name__ == "__main__":
    main()
