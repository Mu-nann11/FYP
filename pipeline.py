from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .files import get_file_pattern, get_image_files
from .stitching import (
    configure_stitching_parameters,
    build_macro_command,
    execute_stitching_with_retry,
)
from .outputs import (
    validate_and_open_result,
    _snapshot_candidates,
)
from .ui import timeout_input

CHANNEL_ORDER = ["DAPI", "HER2", "PR", "ER"]


def run_stitch_for_channel(
    level1,
    channel,
    params,
    config,
    ij,
    logger,
    output_dir,
):
    ch_dir = level1 / channel
    if not ch_dir.is_dir():
        logger.error("Channel dir not found: %s", ch_dir)
        print("❌ 未找到通道目录: %s" % ch_dir)
        return False, None

    pattern = get_file_pattern(str(ch_dir), interactive=config["INTERACTIVE"])
    if not pattern:
        logger.error("No pattern for %s", ch_dir)
        print("❌ %s 下无法推断图像文件模式，跳过 %s" % (ch_dir, channel))
        return False, None

    img_files = get_image_files(str(ch_dir), pattern=pattern)
    if not img_files:
        logger.error("No image files for %s pattern=%s", ch_dir, pattern)
        print("❌ %s 下未找到匹配 %s 的图像文件，跳过 %s" % (ch_dir, pattern, channel))
        return False, None

    logger.info("Channel %s: found %s files (%s)", channel, len(img_files), pattern)
    print("ℹ️ 通道 %s：找到 %s 个匹配文件，开始拼接" % (channel, len(img_files)))

    fused_name = "%s_%s" % (level1.name, channel)
    tile_cfg_name = "TileConfiguration_%s.txt" % fused_name
    before_candidates = _snapshot_candidates(output_dir)

    macro = build_macro_command(
        input_dir=str(ch_dir),
        output_dir=str(output_dir),
        file_pattern=pattern,
        params=params,
        tile_config_name=tile_cfg_name,
    )
    logger.debug("Macro for %s:\n%s", ch_dir, macro)

    ok = execute_stitching_with_retry(ij, macro, logger, output_dir=output_dir, max_retries=3)
    if not ok:
        logger.error("Stitching failed for %s %s", level1.name, channel)
        print("❌ %s 通道 %s 拼接失败" % (level1.name, channel))
        return False, None

    result = validate_and_open_result(
        output_dir,
        config,
        fused_name,
        logger,
        before_candidates,
    )
    if result is None:
        logger.error("No output tiff for %s %s", level1.name, channel)
        print("❌ %s 通道 %s 宏已执行，但未找到输出文件" % (level1.name, channel))
        return False, None

    return True, result


def check_channel_sizes(results, logger):
    try:
        import tifffile
    except ImportError:
        logger.warning("tifffile not installed; skip size check")
        return True

    info = {}
    for ch, p in results.items():
        try:
            arr = tifffile.imread(str(p))
            info[ch] = {"shape": arr.shape, "dtype": str(arr.dtype)}
        except Exception as e:
            logger.exception("Read tiff failed for %s: %s", p, e)

    if len(info) <= 1:
        return True

    shapes = dict((ch, d["shape"]) for ch, d in info.items())
    dtypes = dict((ch, d["dtype"]) for ch, d in info.items())

    shapes_match = len(set(shapes.values())) == 1
    dtypes_match = len(set(dtypes.values())) == 1

    if shapes_match and dtypes_match:
        logger.info("All channels match in shape and dtype: %s / %s", list(shapes.values())[0], list(dtypes.values())[0])
        print("✅ 四个通道输出一致：shape=%s, dtype=%s" % (list(shapes.values())[0], list(dtypes.values())[0]))
        return True

    print("⚠️ 注意：四个通道输出格式不一致，请检查该块拼接结果")
    logger.warning("Channel info differs:")
    for ch, i in info.items():
        logger.warning("  %s: %s", ch, i)
        print("  %s: %s" % (ch, i))
    return False


def process_level1_sequential(level1_path, config, ij, logger):
    level1 = Path(level1_path)
    logger.info("Processing level1 (sequential, multi-channel): %s", level1)

    print("\n" + "=" * 60)
    print("开始处理一级目录: %s" % level1.name)
    print("完整路径: %s" % level1)
    print("=" * 60)

    params = configure_stitching_parameters(config, interactive=config["INTERACTIVE"])

    stitched_parent = Path(config["STITCHED_PARENT_DIR"])
    stitched_parent.mkdir(parents=True, exist_ok=True)
    output_dir = stitched_parent / level1.name
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for ch in CHANNEL_ORDER:
        ok, result = run_stitch_for_channel(
            level1=level1,
            channel=ch,
            params=params,
            config=config,
            ij=ij,
            logger=logger,
            output_dir=output_dir,
        )
        if ok and result is not None:
            results[ch] = result

    if results:
        check_channel_sizes(results, logger)

    print("✅ %s 处理完成，结果位于: %s" % (level1.name, output_dir))
    logger.info("Level1 done (sequential): %s", level1)


def process_all_level1_dirs(level1_dirs, config, ij, logger):
    total = len(level1_dirs)
    for i, p in enumerate(level1_dirs, 1):
        print("\n\n" + "#" * 60)
        print("处理进度: %s/%s" % (i, total))
        print("#" * 60)

        if config["INTERACTIVE"]:
            cont = timeout_input(
                "即将处理 %s。继续? (Y=继续, n=退出, s=跳过，默认Y)" % Path(p).name,
                "Y",
                5,
                True,
            ).strip().lower()
            if cont in ("n", "no"):
                logger.info("User stopped at %s/%s", i, total)
                print("用户选择停止处理，程序退出")
                return
            if cont in ("s", "skip"):
                logger.info("User skipped %s", p)
                print("跳过 %s" % Path(p).name)
                continue

        process_level1_sequential(p, config, ij, logger)

    print("\n🎉 所有一级目录处理完毕！")
    logger.info("All level1 directories processed")
