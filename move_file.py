import os
import shutil

# -------------------------- 配置参数 --------------------------
source_root = r"D:\Munan_FYP\TMA2_BC081116d\TMA2_BC081116d"  # 原始数据路径
target_root = r"\\COMPDrive\Student4\24037101d\COMProfile\Desktop\Raw_Data\TMAd"  # 目标路径
selected_blocks = ["A3", "A10", "C5", "C7", "E10", "G6", "K1", "K10", "B3", "D8"]  # 筛选的10个Block

# 通道映射：Cycle1关键词→目标文件夹
cycle1_channel_map = {
    "w1DAPI": "DAPI",
    "w2GFP": "HER2",
    "w3Cy3": "ER",
    "w4Cy5": "PR"
}
cycle2_target_folder = "Ki67"  # Cycle2文件对应Ki67文件夹
cycle2_file_prefix = "Composite"  # Cycle2文件前缀

# -------------------------- 核心函数 --------------------------
def copy_files_to_target(source_path, target_block, target_channel):
    """按目标层级复制文件：TMAd/Block/Channel/文件名"""
    try:
        target_dir = os.path.join(target_root, target_block, target_channel)
        os.makedirs(target_dir, exist_ok=True)  # 自动创建文件夹
        target_file = os.path.join(target_dir, os.path.basename(source_path))
        shutil.copy2(source_path, target_file)  # 保留文件元信息
        print(f"✅ 成功：{os.path.basename(source_path)} → {target_block}/{target_channel}")
    except Exception as e:
        print(f"❌ 失败 {os.path.basename(source_path)}：{str(e)}")

# -------------------------- 执行抽取 --------------------------
def extract_selected_blocks():
    print("🚀 开始抽取10个测试Block的文件...")
    
    # 1. 处理Cycle1（DAPI/HER2/ER/PR）
    cycle1_source = os.path.join(source_root, "Cycle1")
    if os.path.exists(cycle1_source):
        for block in selected_blocks:
            block_source = os.path.join(cycle1_source, block)
            if not os.path.exists(block_source):
                print(f"⚠️ 跳过：Cycle1中无{block}文件夹")
                continue
            # 遍历Block下所有TIF文件
            for filename in os.listdir(block_source):
                if filename.lower().endswith(".tif"):
                    # 匹配通道并复制
                    for keyword, channel in cycle1_channel_map.items():
                        if keyword in filename:
                            copy_files_to_target(os.path.join(block_source, filename), block, channel)
                            break
    else:
        print(f"❌ 错误：Cycle1路径不存在 → {cycle1_source}")

    # 2. 处理Cycle2（Ki67）
    cycle2_source = os.path.join(source_root, "Cycle2")
    if os.path.exists(cycle2_source):
        for block in selected_blocks:
            block_source = os.path.join(cycle2_source, block)
            if not os.path.exists(block_source):
                print(f"⚠️ 跳过：Cycle2中无{block}文件夹")
                continue
            # 遍历Block下所有Composite文件
            for filename in os.listdir(block_source):
                if filename.lower().endswith(".tif") and cycle2_file_prefix in filename:
                    copy_files_to_target(os.path.join(block_source, filename), block, cycle2_target_folder)
    else:
        print(f"❌ 错误：Cycle2路径不存在 → {cycle2_source}")

    print(f"\n🎉 抽取完成！文件已保存至：{target_root}")

# 运行脚本
if __name__ == "__main__":
    if not os.path.exists(source_root):
        print(f"❌ 原始数据路径不存在：{source_root}")
    else:
        extract_selected_blocks()