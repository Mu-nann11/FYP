import subprocess
import time
from pathlib import Path


def _is_result_candidate(p):
    if p.suffix.lower() in (".tif", ".tiff"):
        return True
    if p.suffix == "" and p.name.startswith("img_t1_z1_c1"):
        return True
    return False


def _list_candidates(dir_path):
    out = []
    if not dir_path.exists():
        return out
    for p in dir_path.iterdir():
        if p.is_file() and _is_result_candidate(p):
            out.append(p)
    out.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return out


def _snapshot_candidates(dir_path):
    return set(_list_candidates(dir_path))


def _pick_newest_new_file(before, after):
    new_files = list(after - before)
    if not new_files:
        return None
    new_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return new_files[0]


def safe_rename(src, desired_path, logger):
    desired_path.parent.mkdir(parents=True, exist_ok=True)

    if desired_path.suffix.lower() not in (".tif", ".tiff"):
        desired_path = desired_path.with_suffix(".tif")

    if desired_path.exists():
        ts = time.strftime("%Y%m%d_%H%M%S")
        desired_path = desired_path.with_name("%s_%s%s" % (desired_path.stem, ts, desired_path.suffix))
        logger.warning("Target exists; using new name: %s", desired_path.name)

    src.rename(desired_path)
    return desired_path


def locate_and_rename_output(output_dir, fused_filename, logger, before_candidates):
    after_candidates = _snapshot_candidates(output_dir)
    newest_new = _pick_newest_new_file(before_candidates, after_candidates)

    if newest_new is None:
        for p in _list_candidates(output_dir):
            if p.name.startswith("img_t1_z1_c1") or p.suffix.lower() in (".tif", ".tiff"):
                newest_new = p
                logger.warning("Fallback to latest candidate: %s", p.name)
                break

    if newest_new is None:
        return None

    target = output_dir / ("%s.tif" % fused_filename)
    try:
        return safe_rename(newest_new, target, logger)
    except Exception as e:
        logger.exception("Rename failed: %s", e)
        return newest_new


def validate_and_open_result(output_dir, config, fused_filename, logger, before_candidates):
    result = locate_and_rename_output(output_dir, fused_filename, logger, before_candidates)
    if result is None:
        logger.error("No output file found in %s", output_dir)
        print("❌ 未找到拼接输出文件（可能宏未生成结果）")
        return None

    try:
        size = result.stat().st_size
        size_mb = size / 1024 / 1024
        print("✅ 找到拼接文件: %s" % result.name)
        print(" 文件大小: %s bytes (%.2f MB)" % (format(size, ","), size_mb))
        print(" 完整路径: %s" % result)
        logger.info("Result: %s (%s bytes)", result, size)
    except Exception as e:
        logger.exception("Stat failed: %s", e)

    if bool(config.get("AUTO_OPEN_RESULT", False)):
        fiji_exe = Path(config["FIJI_EXE"])
        if fiji_exe.exists():
            try:
                subprocess.Popen(
                    [str(fiji_exe), str(result)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                logger.info("Opened in Fiji: %s", result.name)
            except Exception as e:
                logger.exception("Open in Fiji failed: %s", e)
        else:
            logger.error("FIJI_EXE not found: %s", fiji_exe)

    return result


def _list_tiffs_recursively(dir_path):
    out = []
    if not dir_path.exists():
        return out
    for p in dir_path.rglob("*"):
        if p.is_file() and p.suffix.lower() in (".tif", ".tiff"):
            out.append(p)
    out.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return out


def open_all_stitched_results(config, logger):
    from .ui import timeout_input

    target_dir = Path(config["STITCHED_PARENT_DIR"])
    fiji_exe = Path(config["FIJI_EXE"])

    if not target_dir.exists():
        print("❌ 目标路径不存在: %s" % target_dir)
        return
    if not fiji_exe.exists():
        print("❌ Fiji 可执行文件不存在: %s" % fiji_exe)
        return

    tiffs = _list_tiffs_recursively(target_dir)
    if not tiffs:
        print("❌ 未找到任何 TIFF 文件")
        return

    max_open = int(config.get("MAX_OPEN_FILES", 30))
    tiffs = tiffs[:max_open]

    print("将打开 %s 个 TIFF（最多 %s 个）：" % (len(tiffs), max_open))
    for p in tiffs:
        print(" - %s" % p)

    confirm = timeout_input("是否继续打开？(Y/N，默认N)", "N", 5, config["INTERACTIVE"]).strip().lower()
    if confirm not in ("y", "yes"):
        print("已取消打开操作")
        return

    for p in tiffs:
        try:
            subprocess.Popen(
                [str(fiji_exe), str(p)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            logger.info("Opened: %s", p)
            time.sleep(1)
        except Exception as e:
            logger.exception("Open failed: %s", e)


def open_single_stitched_result(config, logger):
    from .ui import timeout_input

    target_dir = Path(config["STITCHED_PARENT_DIR"])
    fiji_exe = Path(config["FIJI_EXE"])

    if not target_dir.exists():
        print("❌ 目标路径不存在: %s" % target_dir)
        return
    if not fiji_exe.exists():
        print("❌ Fiji 可执行文件不存在: %s" % fiji_exe)
        return

    tiffs = _list_tiffs_recursively(target_dir)
    if not tiffs:
        print("❌ 未找到任何 TIFF 文件")
        return

    print("\n可用 TIFF 文件（按最新排序）:")
    for i, p in enumerate(tiffs[:50], 1):
        rel = p.relative_to(target_dir)
        print(" %s. %s" % (i, rel))

    choice = timeout_input(
        "请选择要打开的文件 (1-%s)" % min(50, len(tiffs)),
        "1",
        10,
        config["INTERACTIVE"],
    ).strip() or "1"

    try:
        idx = int(choice) - 1
    except ValueError:
        print("输入无效")
        return

    if idx < 0 or idx >= min(50, len(tiffs)):
        print("输入超出范围")
        return

    p = tiffs[idx]
    try:
        subprocess.Popen(
            [str(fiji_exe), str(p)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        logger.info("Opened single: %s", p)
        print("✅ 已启动 Fiji 打开: %s" % p.name)
    except Exception as e:
        logger.exception("Open single failed: %s", e)
        print("❌ 打开失败: %s" % e)
