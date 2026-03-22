import argparse
import os
import shutil
import traceback
from pathlib import Path
from utils import get_logger
from config import config

logger = get_logger("rename_file")

MARKER_FOLDER_MAP = {
    "w1DAPI": "DAPI",
    "w2GFP": "HER2",
    "w3Cy3": "PR",
    "w4Cy5": "ER",
}


def _resolve_process_root(args) -> Path:
    """
    根据 CLI 参数确定待处理的根目录。
    优先级：--path > --raw-subdir > DEFAULT_ROOT_DIR/Raw_Data
    """
    if args.path:
        p = Path(args.path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"--path 指定的目录不存在: {p}")
        return p

    default_root = config.get("DEFAULT_ROOT_DIR", "/data")
    raw_data_name = config.get("RAW_DATA_DIR_NAME", "Raw_Data")

    if args.dataset and args.cycle:
        p = Path(default_root) / raw_data_name / args.dataset / args.cycle
        if not p.exists():
            raise FileNotFoundError(f"拼出的路径不存在: {p}")
        return p

    if args.raw_subdir:
        p = Path(default_root) / raw_data_name / args.raw_subdir
        if not p.exists():
            raise FileNotFoundError(f"拼出的路径不存在: {p}")
        return p

    return Path(default_root) / raw_data_name


def correct_filename_prefix(folder_path: str):
    """
    修正文件夹内文件的前缀为文件夹名（核心：以文件夹名为标准）
    """
    try:
        folder_name = os.path.basename(folder_path)
        logger.info(f"开始处理文件夹: {folder_name}")

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isdir(file_path):
                continue

            filename_clean = filename.replace("__", "_")
            name_parts = filename_clean.split("_")
            if len(name_parts) < 3:
                logger.warning(f"跳过非标准命名文件: {filename}")
                continue

            old_prefix = name_parts[0]
            if old_prefix == folder_name:
                logger.debug(f"文件名前缀已正确，跳过: {filename}")
                continue

            new_name = f"{folder_name}_{'_'.join(name_parts[1:])}"
            new_file_path = os.path.join(folder_path, new_name)

            if os.path.exists(new_file_path):
                logger.warning(f"新文件名已存在，跳过修正: {filename} → {new_name}")
                continue

            os.rename(file_path, new_file_path)
            logger.info(f"修正文件名: {filename} → {new_name}")
    except Exception as e:
        logger.error(f"修正文件名失败 {folder_path}: {e}")
        logger.error(traceback.format_exc())


def classify_tma_files(source_dir: str, marker_map: dict):
    """
    将修正后的文件按标记归类到子文件夹（修复双下划线问题）
    """
    try:
        folder_name = os.path.basename(source_dir)
        logger.info(f"开始归类 {folder_name} 文件夹文件:")

        for filename in os.listdir(source_dir):
            file_path = os.path.join(source_dir, filename)
            if os.path.isdir(file_path):
                continue

            filename_clean = filename.replace("__", "_")
            name_parts = [part for part in filename_clean.split("_") if part]

            if len(name_parts) < 3:
                logger.warning(f"跳过非标准命名文件: {filename}")
                continue

            marker = name_parts[1]
            if marker not in marker_map:
                logger.warning(f"跳过未知标记文件: {filename} (标记: {marker})")
                continue

            target_folder = os.path.join(source_dir, marker_map[marker])
            os.makedirs(target_folder, exist_ok=True)

            target_path = os.path.join(target_folder, filename)
            if os.path.exists(target_path):
                logger.warning(f"目标文件已存在，跳过归类: {filename}")
                continue

            shutil.move(file_path, target_path)
            logger.info(f"归类完成: {filename} → {marker_map[marker]}/")
    except Exception as e:
        logger.error(f"归类文件失败 {source_dir}: {e}")
        logger.error(traceback.format_exc())


def batch_process_folders(root_dir: str, marker_map: dict):
    """
    批量处理根目录下的所有子文件夹（修正文件名+归类）
    """
    try:
        if not os.path.exists(root_dir):
            logger.error(f"根目录不存在: {root_dir}")
            return

        subfolders = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, f))
        ]

        if not subfolders:
            logger.warning(f"根目录 {root_dir} 下未找到任何子文件夹！")
            return

        logger.info(f"检测到 {len(subfolders)} 个待处理文件夹")

        for folder in subfolders:
            correct_filename_prefix(folder)
            classify_tma_files(folder, marker_map)
            logger.info(f"{os.path.basename(folder)} 处理完成！")
    except Exception as e:
        logger.error(f"批量处理文件夹失败: {e}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TMA 文件重命名与归类")
    parser.add_argument("--path", default=None,
                        help="直接指定待处理的根目录（优先级最高）")
    parser.add_argument("--dataset", default=None, choices=["TMAe", "TMAd"],
                        help="数据集名称，配合 --cycle 使用")
    parser.add_argument("--cycle", default=None,
                        help="Cycle 名称，如 Cycle1 / Cycle2，配合 --dataset 使用")
    parser.add_argument("--raw-subdir", default=None,
                        help="Raw_Data 下的子目录，如 'TMAe/Cycle1'")
    args = parser.parse_args()

    ROOT_FOLDER = str(_resolve_process_root(args))

    logger.info("=" * 60)
    logger.info("开始批量处理（修正文件名+修复双下划线归类）")
    logger.info(f"根目录: {ROOT_FOLDER}")
    logger.info("=" * 60)

    batch_process_folders(ROOT_FOLDER, MARKER_FOLDER_MAP)
    logger.info("所有文件夹处理完成！")
