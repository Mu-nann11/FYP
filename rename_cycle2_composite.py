#!/usr/bin/env python3
"""
重命名 Cycle2 中的 Composite 文件

使用方式：
  python rename_cycle2_composite.py <dataset>
  
例如：
  python rename_cycle2_composite.py TMAd
  python rename_cycle2_composite.py TMAe
"""

import os
import shutil
import re
from pathlib import Path
import argparse


def rename_composite_files(block_path: Path, dataset: str):
    """重命名 Block 下的 Composite 文件"""
    block_name = block_path.name
    
    if not block_path.is_dir():
        return
    
    composite_files = sorted([f for f in block_path.iterdir() if f.is_file() and 'composite' in f.name.lower()])
    
    if not composite_files:
        print(f"  ⚠️ Block {block_name} 下没有找到 Composite 文件")
        return
    
    for file_path in composite_files:
        # 从文件名中提取序号
        # 从 Composite-1.tif 或 Composite_1.tif 等格式提取序号
        match = re.search(r'composite[\s_-](\d+)', file_path.name, re.IGNORECASE)
        if not match:
            print(f"  ⚠️ 无法从 {file_path.name} 提取序号")
            continue
        
        seq_num = match.group(1)
        ext = file_path.suffix
        
        # 新文件名格式：{block}_{dataset}_Composite_{序号}.{扩展名}
        # 例如：A2_TMAd_Composite_1.tif
        new_filename = f"{block_name}_{dataset}_Composite_{seq_num}{ext}"
        dest_path = block_path / new_filename
        
        # 如果目标文件已存在，跳过
        if dest_path.exists():
            print(f"  ⏭️ 跳过（已存在）: {new_filename}")
            continue
        
        try:
            shutil.move(str(file_path), str(dest_path))
            print(f"  ✅ 重命名: {file_path.name} → {new_filename}")
        except Exception as e:
            print(f"  ❌ 失败: {file_path.name} - {e}")


def rename_cycle2(dataset_root: Path, dataset: str):
    """重命名整个 Cycle2 下所有 Block 的 Composite 文件"""
    cycle_path = dataset_root / dataset / "Cycle2"
    
    if not cycle_path.exists():
        print(f"❌ 路径不存在: {cycle_path}")
        return
    
    print(f"📁 开始重命名 {dataset}/Cycle2/ 下的 Composite 文件")
    print(f"   路径: {cycle_path}\n")
    
    block_dirs = sorted([d for d in cycle_path.iterdir() if d.is_dir()])
    
    if not block_dirs:
        print(f"❌ 未找到任何 Block 目录")
        return
    
    for i, block_path in enumerate(block_dirs, 1):
        block_name = block_path.name
        print(f"[{i}/{len(block_dirs)}] 处理 Block: {block_name}")
        rename_composite_files(block_path, dataset)
        print()
    
    print(f"✅ {dataset}/Cycle2/ 重命名完成！")
    print(f"   新格式: {{block}}_{{dataset}}_Composite_{{序号}}.tif")
    print(f"   例如: A2_{dataset}_Composite_1.tif")


def main():
    parser = argparse.ArgumentParser(
        description="重命名 Cycle2 中的 Composite 文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python rename_cycle2_composite.py TMAd
  python rename_cycle2_composite.py TMAe

完成后的文件格式：
  {block}_{dataset}_Composite_{序号}.tif
  例如：A2_TMAd_Composite_1.tif
        """
    )
    
    parser.add_argument("dataset", help="数据集名称 (TMAd 或 TMAe)")
    parser.add_argument("--root", default="data/Raw_Data", help="Raw_Data 根目录（默认: %(default)s）")
    
    args = parser.parse_args()
    
    root_path = Path(args.root)
    if not root_path.exists():
        print(f"❌ Raw_Data 根目录不存在: {root_path}")
        return
    
    rename_cycle2(root_path, args.dataset)


if __name__ == "__main__":
    main()
