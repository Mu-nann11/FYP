import time
from pathlib import Path

from imagej import init as imagej_init


def init_imagej(config):
    return imagej_init(config["FIJI_PATH"], headless=True)


def configure_stitching_parameters(config, interactive):
    from .ui import timeout_input

    params = {
        "fusion_method": str(config.get("DEFAULT_FUSION_METHOD", "Linear Blending")),
        "regression_threshold": str(config.get("DEFAULT_REGRESSION_THRESHOLD", "0.30")),
        "max_displacement": str(config.get("DEFAULT_MAX_DISPLACEMENT", "2.50")),
        "absolute_displacement": str(config.get("DEFAULT_ABSOLUTE_DISPLACEMENT", "3.50")),
        "computation_mode": str(config.get("DEFAULT_COMPUTATION_MODE", "Save memory (but be slower)")),
        "image_output": str(config.get("DEFAULT_IMAGE_OUTPUT", "Write to disk")),
    }

    if not interactive or bool(config.get("AUTO_USE_DEFAULT_PARAMS", True)):
        return params

    print("\n" + "=" * 50)
    print("拼接参数配置")
    print("=" * 50)

    change = timeout_input("是否调整拼接参数？(y/N)", "N", 5, True).strip().lower()
    if change not in ("y", "yes"):
        return params

    print("\n请选择融合方法:")
    print(" 1. Linear Blending")
    print(" 2. Max Intensity")
    print(" 3. Average")
    print(" 4. None")
    fusion_choice = timeout_input("请输入选择 [1]", "1", 5, True).strip() or "1"
    fusion_map = {"1": "Linear Blending", "2": "Max Intensity", "3": "Average", "4": "None"}
    params["fusion_method"] = fusion_map.get(fusion_choice, "Linear Blending")

    params["regression_threshold"] = timeout_input(
        "回归阈值(0-1) [%s]" % params["regression_threshold"],
        params["regression_threshold"],
        5,
        True,
    ).strip() or params["regression_threshold"]

    params["max_displacement"] = timeout_input(
        "最大/平均位移阈值 [%s]" % params["max_displacement"],
        params["max_displacement"],
        5,
        True,
    ).strip() or params["max_displacement"]

    params["absolute_displacement"] = timeout_input(
        "绝对位移阈值 [%s]" % params["absolute_displacement"],
        params["absolute_displacement"],
        5,
        True,
    ).strip() or params["absolute_displacement"]

    print("\n请选择计算模式:")
    print(" 1. Save memory (but be slower)")
    print(" 2. Fast (but use more memory)")
    compute_choice = timeout_input("请输入选择 [1]", "1", 5, True).strip() or "1"
    if compute_choice == "2":
        params["computation_mode"] = "Fast (but use more memory)"

    return params


def build_macro_command(
    input_dir,
    output_dir,
    file_pattern,
    params,
    tile_config_name="TileConfiguration.txt",
):
    input_dir_ij = str(Path(input_dir).resolve()).replace("\\", "/")
    output_dir_ij = str(Path(output_dir).resolve()).replace("\\", "/")

    return """
run("Grid/Collection stitching",
    "type=[Sequential Images] " +
    "order=[All files in directory] " +
    "directory=[%s] " +
    "file_names=[%s] " +
    "output_textfile_name=%s " +
    "fusion_method=[%s] " +
    "regression_threshold=%s " +
    "max/avg_displacement_threshold=%s " +
    "absolute_displacement_threshold=%s " +
    "frame_range_to_compare=1 " +
    "computation_parameters=[%s] " +
    "image_output=[%s] " +
    "output_directory=[%s]");
""" % (
        input_dir_ij,
        file_pattern,
        tile_config_name,
        params["fusion_method"],
        params["regression_threshold"],
        params["max_displacement"],
        params["absolute_displacement"],
        params["computation_mode"],
        params["image_output"],
        output_dir_ij,
    )


def execute_stitching_with_retry(ij, macro_cmd, logger, output_dir=None, max_retries=3):
    for attempt in range(max_retries):
        try:
            logger.info("Running stitching macro attempt %s/%s", attempt + 1, max_retries)
            ij.py.run_macro(macro_cmd)
            logger.info("Stitching macro finished")
            return True
        except Exception as e:
            logger.exception("Stitching failed attempt %s/%s: %s", attempt + 1, max_retries, e)
            if attempt < max_retries - 1:
                wait = (attempt + 1) * 10
                logger.info("Wait %s seconds before retry", wait)
                time.sleep(wait)
            else:
                if output_dir is not None:
                    try:
                        for f in Path(output_dir).glob("TileConfiguration_*.txt"):
                            f.unlink(missing_ok=True)
                    except Exception:
                        pass
                return False
    return False
