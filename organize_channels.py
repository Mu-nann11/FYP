#!/usr/bin/env python3
"""
组织 Raw_Data 中的图像文件按通道分类到对应的子目录

使用方式：
  python organize_channels.py <dataset> <cycle>
  
例如：
  python organize_channels.py TMAd Cycle1
  python organize_channels.py TMAe Cycle1
"""

import os
import shutil
import re
from pathlib import Path
import argparse

# 通道映射：文件名关键词 → 目标目录名
CHANNEL_MAPPING = {
    "Cycle1": {
        "w1DAPI": "DAPI",
        "w1dapi": "DAPI",
        "w2GFP": "HER2",
        "w2gfp": "HER2",
        "w3Cy3": "PR",
        "w3cy3": "PR",
        "w4Cy5": "ER",
        "w4cy5": "ER",
    },
    "Cycle2": {
        "Composite": "Composite",
    }
}


def get_channel_keyword(filename: str, cycle: str = "Cycle1") -> tuple:
    """从文件名提取通道关键字和通道名称
    
    返回: (通道名称, 通道关键字, 序号)
    例如: ("DAPI", "w1DAPI", "1") 或 ("HER2", "w2GFP", "1")
    """
    filename_upper = str(filename).upper()
    
    for keyword, channel_name in CHANNEL_MAPPING.get(cycle, {}).items():
        if keyword.upper() in filename_upper:
            # 从文件名中提取序号（s后面的数字）
            match = re.search(r'_s(\d+)', filename, re.IGNORECASE)
            seq_num = match.group(1) if match else ""
            return channel_name, keyword, seq_num
    
    return None, None, None


def organize_block(block_path: Path, dataset: str, cycle: str = "Cycle1"):
    """组织单个 Block 下的文件到通道子目录，并重命名文件"""
    block_name = block_path.name
    
    if not block_path.is_dir():
        return
    
    # 收集按通道分类的文件
    channel_files = {}
    
    for filename in block_path.iterdir():
        if not filename.is_file():
            continue
        
        channel, keyword, seq_num = get_channel_keyword(filename.name, cycle)
        if not channel:
            print(f"  ⚠️ 无法识别通道: {filename.name}")
            continue
        
        if channel not in channel_files:
            channel_files[channel] = []
        channel_files[channel].append((filename, keyword, seq_num))
    
    if not channel_files:
        print(f"  ⚠️ Block {block_name} 下没有找到对应的通道文件")
        return
    
    # 创建通道子目录并移动文件
    for channel, files in sorted(channel_files.items()):
        channel_dir = block_path / channel
        channel_dir.mkdir(exist_ok=True)
        
        for file_path, keyword, seq_num in files:
            # 新文件名格式：{block_name}_{dataset}_{keyword}_s{序号}.{扩展名}
            # 例如：A10_TMAd_w1DAPI_s1.TIF
            ext = file_path.suffix
            new_filename = f"{block_name}_{dataset}_{keyword}_s{seq_num}{ext}"
            dest_path = channel_dir / new_filename
            
            # 如果目标文件已存在，跳过
            if dest_path.exists():
                print(f"  ⏭️ 跳过（已存在）: {channel}/{new_filename}")
                continue
            
            try:
                shutil.move(str(file_path), str(dest_path))
                print(f"  ✅ 移动并重命名: {file_path.name} → {channel}/{new_filename}")
            except Exception as e:
                print(f"  ❌ 失败: {file_path.name} - {e}")


def organize_cycle(dataset_root: Path, dataset: str, cycle: str):
    """组织整个 Cycle 下所有 Block 的文件"""
    cycle_path = dataset_root / dataset / cycle
    
    if not cycle_path.exists():
        print(f"❌ 路径不存在: {cycle_path}")
        return
    
    print(f"📁 开始组织 {dataset}/{cycle}/ 下的数据")
    print(f"   路径: {cycle_path}\n")
    
    block_dirs = sorted([d for d in cycle_path.iterdir() if d.is_dir()])
    
    if not block_dirs:
        print(f"❌ 未找到任何 Block 目录")
        return
    
    for i, block_path in enumerate(block_dirs, 1):
        block_name = block_path.name
        print(f"[{i}/{len(block_dirs)}] 处理 Block: {block_name}")
        organize_block(block_path, dataset, cycle)
        print()
    
    print(f"✅ {dataset}/{cycle}/ 组织完成！")


def main():
    parser = argparse.ArgumentParser(
        description="组织原始数据按通道分类到子目录",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python organize_channels.py TMAd Cycle1
  python organize_channels.py TMAe Cycle1
  python organize_channels.py TMAd Cycle2
        """
    )
    
    parser.add_argument("dataset", help="数据集名称 (TMAd 或 TMAe)")
    parser.add_argument("cycle", help="Cycle 名称 (Cycle1 或 Cycle2)")
    parser.add_argument("--root", default="data/Raw_Data", help="Raw_Data 根目录默认: %(default)s)")
    
    args = parser.parse_args()
    
    root_path = Path(args.root)
    if not root_path.exists():
        print(f"❌ Raw_Data 根目录不存在: {root_path}")
        return
    
    organize_cycle(root_path, args.dataset, args.cycle)


if __name__ == "__main__":
    main()
