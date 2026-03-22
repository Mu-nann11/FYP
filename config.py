import copy
import datetime
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def _project_root():
    return Path(__file__).resolve().parent.parent


def _load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _truthy(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


def _as_float(v, default, min_v=None, max_v=None):
    try:
        x = float(v)
    except Exception:
        x = float(default)
    if min_v is not None and x < min_v:
        x = min_v
    if max_v is not None and x > max_v:
        x = max_v
    return x


def _as_int(v, default, min_v=None, max_v=None):
    try:
        x = int(v)
    except Exception:
        x = int(default)
    if min_v is not None and x < min_v:
        x = min_v
    if max_v is not None and x > max_v:
        x = max_v
    return x


def _as_float_str(v, default, min_v=0.0, max_v=None):
    x = _as_float(v, default, min_v=min_v, max_v=max_v)
    return "%.2f" % x


def _normalize_keys(cfg):
    alias = {
        "FIJIPATH": "FIJI_PATH",
        "FIJIEXE": "FIJI_EXE",
        "STITCHEDPARENTDIR": "STITCHED_PARENT_DIR",
        "AUTOOPENRESULT": "AUTO_OPEN_RESULT",
        "AUTOUSEDEFAULTPARAMS": "AUTO_USE_DEFAULT_PARAMS",
        "DEFAULTFUSIONMETHOD": "DEFAULT_FUSION_METHOD",
        "DEFAULTREGRESSIONTHRESHOLD": "DEFAULT_REGRESSION_THRESHOLD",
        "DEFAULTMAXDISPLACEMENT": "DEFAULT_MAX_DISPLACEMENT",
        "DEFAULTABSOLUTEDISPLACEMENT": "DEFAULT_ABSOLUTE_DISPLACEMENT",
        "DEFAULTCOMPUTATIONMODE": "DEFAULT_COMPUTATION_MODE",
        "DEFAULTIMAGEOUTPUT": "DEFAULT_IMAGE_OUTPUT",
        "DEFAULT_image_output": "DEFAULT_IMAGE_OUTPUT",
        "LOGLEVEL": "LOG_LEVEL",
    }
    out = dict(cfg)
    for k_old, k_new in alias.items():
        if k_old in out and k_new not in out:
            out[k_new] = out[k_old]
    return out


def _deep_merge(base, extra):
    out = copy.deepcopy(base)
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _resolve_path(val):
    if not val:
        return val
    expanded = os.path.expandvars(os.path.expanduser(str(val).strip()))
    if os.path.isabs(expanded):
        return str(Path(expanded).resolve())
    return str((_project_root() / expanded).resolve())


def load_config(config_path=None):
    defaults = {
        "FIJI_PATH": "/opt/Fiji.app",
        "FIJI_EXE": "/opt/Fiji.app/ImageJ-linux64",
        "STITCHED_PARENT_DIR": "/results/stitched",
        "DEFAULT_ROOT_DIR": "/data",
        "RAW_DATA_DIR_NAME": "Raw_Data",
        "AUTO_OPEN_RESULT": False,
        "AUTO_USE_DEFAULT_PARAMS": True,
        "DEFAULT_FUSION_METHOD": "Linear Blending",
        "DEFAULT_REGRESSION_THRESHOLD": "0.30",
        "DEFAULT_MAX_DISPLACEMENT": "2.50",
        "DEFAULT_ABSOLUTE_DISPLACEMENT": "3.50",
        "DEFAULT_COMPUTATION_MODE": "Save memory (but be slower)",
        "DEFAULT_IMAGE_OUTPUT": "Write to disk",
        "LOG_LEVEL": "INFO",
        "INTERACTIVE": False,
        "MAX_OPEN_FILES": 30,

        "CROP_INPUT_DIR": "/results/stitched",
        "CROP_OUTPUT_DIR": "/results/crop",
        "CROP_MARGIN": 20,

        "BATCH_OUTPUT_DIR": "/results/batch_features",
        "BATCH_LOG_CSV": "batch_log.csv",
        "BATCH_RESULT_CSV": "all_blocks_cell_features.csv",
        "BATCH_STATE_CSV": "batch_state.csv",
        "BATCH_OVERLAY_DIR": "overlays",
        "BATCH_RESUME": True,
        "BATCH_SAVE_OVERLAY": True,
        "BATCH_DO_ALIGN": True,
        "BATCH_SKIP_DONE": True,

        "FEATURE_OUTPUT_DIR": "/results/compared_result",

        "STITCH_ALLOW_CYCLE2_COMPOSITE": False,

        "SEGMENTATION": {
            "MODEL_TYPE": "nuclei",
            "USE_GPU": False,
            "DIAMETER": None,
            "FLOW_THRESHOLD": 0.4,
            "CELLPROB_THRESHOLD": 0.0,
            "CHANNELS": [0, 0],
        },

        "LOADER": {
            "CHANNELS": ["DAPI", "HER2", "PR", "ER"],
            "CYCLE2_CHANNELS": ["DAPI", "KI67"],
            "DO_PREPROCESS": False,
            "CLAHE_CLIP_LIMIT": 2.0,
            "CLAHE_TILE_GRID_SIZE": [8, 8],
            "GAUSSIAN_BLUR_KERNEL": [3, 3],
        },
    }

    if config_path is None:
        config_path = str(_project_root() / "fiji_config.json")

    cfg_file = Path(config_path)
    user_cfg = _load_json(cfg_file) if cfg_file.exists() else {}
    cfg = _deep_merge(defaults, _normalize_keys(user_cfg))

    cfg["AUTO_OPEN_RESULT"] = _truthy(cfg.get("AUTO_OPEN_RESULT", False))
    cfg["AUTO_USE_DEFAULT_PARAMS"] = _truthy(cfg.get("AUTO_USE_DEFAULT_PARAMS", True))
    cfg["INTERACTIVE"] = _truthy(cfg.get("INTERACTIVE", False))
    cfg["BATCH_RESUME"] = _truthy(cfg.get("BATCH_RESUME", True))
    cfg["BATCH_SAVE_OVERLAY"] = _truthy(cfg.get("BATCH_SAVE_OVERLAY", True))
    cfg["BATCH_DO_ALIGN"] = _truthy(cfg.get("BATCH_DO_ALIGN", True))
    cfg["BATCH_SKIP_DONE"] = _truthy(cfg.get("BATCH_SKIP_DONE", True))

    cfg["MAX_OPEN_FILES"] = _as_int(cfg.get("MAX_OPEN_FILES", 30), 30, 1)
    cfg["CROP_MARGIN"] = _as_int(cfg.get("CROP_MARGIN", 20), 20, 0)

    cfg["DEFAULT_REGRESSION_THRESHOLD"] = _as_float_str(cfg.get("DEFAULT_REGRESSION_THRESHOLD"), 0.30, 0.0, 1.0)
    cfg["DEFAULT_MAX_DISPLACEMENT"] = _as_float_str(cfg.get("DEFAULT_MAX_DISPLACEMENT"), 2.50, 0.0, None)
    cfg["DEFAULT_ABSOLUTE_DISPLACEMENT"] = _as_float_str(cfg.get("DEFAULT_ABSOLUTE_DISPLACEMENT"), 3.50, 0.0, None)

    seg = cfg.setdefault("SEGMENTATION", {})
    seg["USE_GPU"] = _truthy(seg.get("USE_GPU", False))
    seg["FLOW_THRESHOLD"] = _as_float(seg.get("FLOW_THRESHOLD", 0.4), 0.4)
    seg["CELLPROB_THRESHOLD"] = _as_float(seg.get("CELLPROB_THRESHOLD", 0.0), 0.0)
    if seg.get("DIAMETER") in (None, "", "None"):
        seg["DIAMETER"] = None
    else:
        seg["DIAMETER"] = _as_float(seg.get("DIAMETER"), 30.0, 0.0)
    if not isinstance(seg.get("CHANNELS"), list) or len(seg["CHANNELS"]) != 2:
        seg["CHANNELS"] = [0, 0]

    loader = cfg.setdefault("LOADER", {})
    loader["DO_PREPROCESS"] = _truthy(loader.get("DO_PREPROCESS", False))
    loader["CLAHE_CLIP_LIMIT"] = _as_float(loader.get("CLAHE_CLIP_LIMIT", 2.0), 2.0, 0.0)

    tile = loader.get("CLAHE_TILE_GRID_SIZE", [8, 8])
    if not isinstance(tile, list) or len(tile) != 2:
        tile = [8, 8]
    loader["CLAHE_TILE_GRID_SIZE"] = [max(1, int(tile[0])), max(1, int(tile[1]))]

    kernel = loader.get("GAUSSIAN_BLUR_KERNEL", [3, 3])
    if not isinstance(kernel, list) or len(kernel) != 2:
        kernel = [3, 3]
    kx = max(1, int(kernel[0]))
    ky = max(1, int(kernel[1]))
    if kx % 2 == 0:
        kx += 1
    if ky % 2 == 0:
        ky += 1
    loader["GAUSSIAN_BLUR_KERNEL"] = [kx, ky]

    for key in (
        "FIJI_PATH",
        "FIJI_EXE",
        "STITCHED_PARENT_DIR",
        "DEFAULT_ROOT_DIR",
        "CROP_INPUT_DIR",
        "CROP_OUTPUT_DIR",
        "BATCH_OUTPUT_DIR",
        "FEATURE_OUTPUT_DIR",
    ):
        cfg[key] = _resolve_path(cfg.get(key, ""))

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg["RUN_LOG_FILENAME"] = "run_%s.log" % ts

    Path(cfg["STITCHED_PARENT_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["CROP_OUTPUT_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["BATCH_OUTPUT_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["FEATURE_OUTPUT_DIR"]).mkdir(parents=True, exist_ok=True)
    (Path(cfg["BATCH_OUTPUT_DIR"]) / cfg["BATCH_OVERLAY_DIR"]).mkdir(parents=True, exist_ok=True)

    return cfg


def apply_cli_overrides(config):
    cfg = copy.deepcopy(config)
    argv = list(sys.argv[1:])
    args = set(argv)

    if "--batch" in args:
        cfg["INTERACTIVE"] = False
    if "--interactive" in args:
        cfg["INTERACTIVE"] = True
    if "--resume" in args:
        cfg["BATCH_RESUME"] = True
    if "--no-resume" in args:
        cfg["BATCH_RESUME"] = False
    if "--gpu" in args:
        cfg["SEGMENTATION"]["USE_GPU"] = True
    if "--cpu" in args:
        cfg["SEGMENTATION"]["USE_GPU"] = False

    def _get_value(flag: str):
        if flag in argv:
            i = argv.index(flag)
            if i + 1 < len(argv):
                return argv[i + 1]
        for a in argv:
            if a.startswith(flag + "="):
                return a.split("=", 1)[1]
        return None

    channels_str = _get_value("--channels")
    if channels_str:
        parts = [p.strip() for p in str(channels_str).split(",")]
        parts = [p for p in parts if p]
        if parts:
            cfg.setdefault("LOADER", {})
            cfg["LOADER"]["CHANNELS"] = parts

    level1 = _get_value("--level1")
    if level1:
        cfg["ONLY_LEVEL1"] = str(level1).strip()

    ref_channel = _get_value("--ref-channel")
    if ref_channel:
        cfg["STITCH_REFERENCE_CHANNEL"] = str(ref_channel).strip()

    return cfg
