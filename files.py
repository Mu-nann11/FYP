import glob
import os
import re

from .ui import timeout_input

SUPPORTED_FORMATS = (".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp", ".lsm", ".czi", ".nd2")


def get_image_files(input_dir, pattern=None, exclude_prefixes=None):
    if exclude_prefixes is None:
        exclude_prefixes = ["img_t1_z1_c1", "stitched", "fused", "TileConfiguration"]

    files = []
    if pattern:
        files.extend(glob.glob(os.path.join(input_dir, pattern), recursive=False))

    for ext in SUPPORTED_FORMATS:
        files.extend(glob.glob(os.path.join(input_dir, "*%s" % ext), recursive=False))
        files.extend(glob.glob(os.path.join(input_dir, "*%s" % ext.upper()), recursive=False))

    out = []
    for f in files:
        name = os.path.basename(f)
        if not any(name.startswith(p) for p in exclude_prefixes):
            out.append(f)

    return sorted(set(out))


def infer_pattern_from_files(input_dir):
    img_files = get_image_files(input_dir)
    if not img_files:
        return None

    filenames = [os.path.basename(f) for f in img_files]
    sample = filenames[0]
    ext = os.path.splitext(sample)[1]

    if re.search(r"\d+", sample):
        return re.sub(r"\d+", "*", sample)

    common_prefix = os.path.commonprefix(filenames)
    if common_prefix and len(common_prefix) > 2:
        return "%s*%s" % (common_prefix, ext)

    return "*%s" % ext


def get_file_pattern(input_dir, interactive):
    print("开始检测文件模式...")

    if interactive:
        user_pattern = timeout_input(
            "请输入待拼接文件匹配 Pattern（如 R6__w1DAPI_s1*.TIF；回车自动检测）",
            default="",
            timeout=5,
            interactive=True,
        ).strip()
        if user_pattern:
            print("用户输入模式: %s" % user_pattern)
            return user_pattern

    pattern = infer_pattern_from_files(input_dir)
    if not pattern:
        print("❌ 未找到支持的图像文件。支持格式: %s" % (", ".join(SUPPORTED_FORMATS)))
        return None

    print("ℹ️ 自动检测到文件模式: %s" % pattern)

    if interactive:
        confirm = timeout_input("是否使用该模式？(Y/n，默认Y)", "Y", 5, True).strip().lower()
        if confirm in ("n", "no"):
            manual = input("请手动输入 Pattern: ").strip()
            return manual if manual else None

    return pattern
