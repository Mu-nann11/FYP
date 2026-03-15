from pathlib import Path
import traceback
import pandas as pd

from loader import CROP_ROOT, load_block
from alignment import align_by_shift
from segmentation import segment_nuclei, save_nuclei_overlay
from features import extract_features

# 容器内输出目录，对应宿主机 ./results/batch_features
OUT_DIR = Path("/results/batch_features")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "all_blocks_cell_features.csv"
LOG_CSV = OUT_DIR / "batch_log.csv"
OVERLAY_DIR = OUT_DIR / "overlays"
OVERLAY_DIR.mkdir(parents=True, exist_ok=True)


def list_blocks():
    if not CROP_ROOT.exists():
        raise FileNotFoundError(f"裁剪结果根目录不存在: {CROP_ROOT}")
    return sorted([p.name for p in CROP_ROOT.iterdir() if p.is_dir()])


def process_one_block(
    block_name: str,
    do_align: bool = True,
    save_overlay: bool = True
) -> pd.DataFrame:
    imgs = load_block(block_name, do_preprocess=True)

    if do_align:
        dapi = imgs["DAPI"]
        for ch in ["HER2", "PR", "ER"]:
            aligned, _, _ = align_by_shift(dapi, imgs[ch])
            imgs[ch] = aligned

    masks = segment_nuclei(imgs["DAPI"])

    if save_overlay:
        out_tif = OVERLAY_DIR / f"{block_name}_nuclei_overlay.tif"
        save_nuclei_overlay(imgs["DAPI"], masks, out_tif)

    df = extract_features(block_name, masks, imgs)

    if "global_cell_id" not in df.columns:
        df["global_cell_id"] = df["block"].astype(str) + "_" + df["cell_id"].astype(str)

    return df


if __name__ == "__main__":
    all_dfs = []
    logs = []

    blocks = list_blocks()
    print("Found blocks:", blocks)

    for b in blocks:
        try:
            df_b = process_one_block(b, do_align=True, save_overlay=True)
            all_dfs.append(df_b)

            logs.append({
                "block": b,
                "status": "OK",
                "n_cells": int(len(df_b)),
                "error": ""
            })
            print(f"[OK] {b}: {len(df_b)} cells")

        except Exception as e:
            logs.append({
                "block": b,
                "status": "FAIL",
                "n_cells": 0,
                "error": repr(e)
            })
            print(f"[FAIL] {b}")
            traceback.print_exc()

    pd.DataFrame(logs).to_csv(LOG_CSV, index=False, encoding="utf-8-sig")
    print(f"Saved log: {LOG_CSV}")

    if len(all_dfs) == 0:
        raise RuntimeError("No blocks processed successfully, nothing to save.")

    df_all = pd.concat(all_dfs, ignore_index=True)

    front_cols = ["global_cell_id", "block", "cell_id"]
    cols = front_cols + [c for c in df_all.columns if c not in front_cols]
    df_all = df_all[cols]

    df_all.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Saved total CSV: {OUT_CSV}")
    print("Total cells:", len(df_all))
    print(f"Saved overlays to: {OVERLAY_DIR}")
