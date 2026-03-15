import os
import shutil

def correct_filename_prefix(folder_path: str):
    """
    修正文件夹内文件的前缀为文件夹名（核心：以文件夹名为标准）
    :param folder_path: 目标文件夹路径
    """
    folder_name = os.path.basename(folder_path)
    print(f"\n🔍 开始处理文件夹: {folder_name}")
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isdir(file_path):
            continue
        
        # 先把双下划线替换成单下划线，统一格式
        filename_clean = filename.replace("__", "_")
        name_parts = filename_clean.split("_")
        if len(name_parts) < 3:
            print(f"⚠️  跳过非标准命名文件: {filename}")
            continue
        
        old_prefix = name_parts[0]
        if old_prefix == folder_name:
            print(f"✅  文件名前缀已正确，跳过: {filename}")
            continue
        
        new_name = f"{folder_name}_{'_'.join(name_parts[1:])}"
        new_file_path = os.path.join(folder_path, new_name)
        
        if os.path.exists(new_file_path):
            print(f"⚠️  新文件名已存在，跳过修正: {filename} → {new_name}")
            continue
        
        os.rename(file_path, new_file_path)
        print(f"🔄  修正文件名: {filename} → {new_name}")

def classify_tma_files(source_dir: str, marker_map: dict):
    """
    将修正后的文件按标记归类到子文件夹（修复双下划线问题）
    :param source_dir: 目标文件夹路径
    :param marker_map: 标记→子文件夹映射
    """
    folder_name = os.path.basename(source_dir)
    print(f"\n📦 开始归类 {folder_name} 文件夹文件:")
    
    for filename in os.listdir(source_dir):
        file_path = os.path.join(source_dir, filename)
        if os.path.isdir(file_path):
            continue
        
        # 关键修复：先把双下划线替换成单下划线，再分割
        filename_clean = filename.replace("__", "_")
        name_parts = [part for part in filename_clean.split("_") if part]  # 过滤空字符串
        
        if len(name_parts) < 3:
            print(f"⚠️  跳过非标准命名文件: {filename}")
            continue
        
        marker = name_parts[1]
        if marker not in marker_map:
            print(f"⚠️  跳过未知标记文件: {filename} (标记: {marker})")
            continue
        
        target_folder = os.path.join(source_dir, marker_map[marker])
        os.makedirs(target_folder, exist_ok=True)
        
        target_path = os.path.join(target_folder, filename)
        if os.path.exists(target_path):
            print(f"⚠️  目标文件已存在，跳过归类: {filename}")
            continue
        
        shutil.move(file_path, target_path)
        print(f"✅  归类完成: {filename} → {marker_map[marker]}/")

def batch_process_folders(root_dir: str, marker_map: dict):
    """
    批量处理根目录下的所有子文件夹（修正文件名+归类）
    :param root_dir: 根目录
    :param marker_map: 标记→子文件夹映射
    """
    subfolders = [
        os.path.join(root_dir, f) 
        for f in os.listdir(root_dir) 
        if os.path.isdir(os.path.join(root_dir, f))
    ]
    
    if not subfolders:
        print("❌ 根目录下未找到任何子文件夹！")
        return
    
    print(f"📁 检测到 {len(subfolders)} 个待处理文件夹:")
    for idx, folder in enumerate(subfolders, 1):
        print(f"   {idx}. {os.path.basename(folder)}")
    
    for folder in subfolders:
        correct_filename_prefix(folder)
        classify_tma_files(folder, marker_map)
        print(f"\n✅✅ {os.path.basename(folder)} 处理完成！\n" + "-"*50)

if __name__ == "__main__":
    ROOT_FOLDER = r"D:\24_12_FYP_1\Code\data\Raw_Data"
    MARKER_FOLDER_MAP = {
        "w1DAPI": "DAPI",
        "w2GFP": "HER2",
        "w3Cy3": "PR",
        "w4Cy5": "ER"
    }
    
    print("="*60)
    print("🎯 开始批量处理（修正文件名+修复双下划线归类）")
    print(f"📌 根目录: {ROOT_FOLDER}")
    print("="*60)
    
    batch_process_folders(ROOT_FOLDER, MARKER_FOLDER_MAP)
    print("\n🎉🎉 所有文件夹处理完成！")