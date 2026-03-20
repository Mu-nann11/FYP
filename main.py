from fiji_stitcher.config import load_config, apply_cli_overrides
from fiji_stitcher.logutil import get_logger
from fiji_stitcher.discovery import get_all_level1_directories
from fiji_stitcher.pipeline import process_all_level1_dirs
from fiji_stitcher.outputs import open_single_stitched_result, open_all_stitched_results
from fiji_stitcher.stitching import init_imagej
from fiji_stitcher.ui import timeout_input

from crop_stitched_results import crop_all_blocks


def main():
    config = apply_cli_overrides(load_config())
    logger = get_logger(config)

    logger.info("Program started")
    logger.info("Initializing ImageJ...")

    try:
        ij = init_imagej(config)
        logger.info("ImageJ initialized")
    except Exception as e:
        logger.exception("ImageJ init failed: %s", e)
        print(f"❌ ImageJ 初始化失败: {e}")
        return

    print("\n" + "=" * 50)
    print("请选择功能:")
    print("1. 处理所有一级目录（批量拼接）")
    print("2. 打开单个拼接结果")
    print("3. 批量打开所有拼接结果")
    print("4. 批量裁剪拼接结果（保持原始位深）")
    print("=" * 50)

    choice = timeout_input(
        "请输入功能编号 (1-4，默认1)",
        default="1",
        timeout=10,
        interactive=config["INTERACTIVE"],
    ).strip() or "1"

    if choice == "1":
        if config.get("ONLY_LEVEL1"):
            level1_dirs = [config["ONLY_LEVEL1"]]
        else:
            level1_dirs = get_all_level1_directories(config)
        if not level1_dirs:
            logger.error("No level1 directories found; exit.")
            print("❌ 无可用一级目录，程序退出")
            return
        process_all_level1_dirs(level1_dirs, config, ij, logger)
        return

    if choice == "2":
        open_single_stitched_result(config, logger)
        return

    if choice == "3":
        open_all_stitched_results(config, logger)
        return

    if choice == "4":
        crop_all_blocks(config)
        print(f"✅ 裁剪完成，输出目录: {config['CROP_OUTPUT_DIR']}")
        return

    print("输入无效，程序退出")


if __name__ == "__main__":
    main()
