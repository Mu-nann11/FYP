import argparse
import traceback
from pathlib import Path
from typing import Optional, List

import pandas as pd

from loader import load_block
from alignment import align_by_shift
from segmentation import segment_nuclei_by_method, get_cytoplasm_masks, save_nuclei_overlay
from features import extract_features, score_markers
from utils import get_logger
from config import config

# 获取统一日志记录器
logger = get_logger("batch_run", log_file=config.batch_output_dir / "batch_run.log")

class BlockProcessor:
    def __init__(
        self,
        seg_method: str = "cellpose",
        do_align: bool = True,
        save_overlay: bool = True,
        overlay_dir: Optional[Path] = None,
        expansion_distance: int = 15
    ):
        self.seg_method = seg_method
        self.do_align = do_align
        self.save_overlay = save_overlay
        self.overlay_dir = overlay_dir or (config.batch_output_dir / "overlays")
        self.expansion_distance = expansion_distance
        
        if self.save_overlay:
            self.overlay_dir.mkdir(parents=True, exist_ok=True)

    def process(self, block_name: str) -> pd.DataFrame:
        logger.info(f"Processing block: {block_name}")
        
        # 1. 加载数据
        imgs = load_block(block_name, do_preprocess=True)

        # 2. 对齐
        if self.do_align:
            dapi = imgs["DAPI"]
            for ch in ["HER2", "PR", "ER"]:
                if ch in imgs:
                    aligned, _, _ = align_by_shift(dapi, imgs[ch])
                    imgs[ch] = aligned
                else:
                    logger.warning(f"Channel {ch} not found in block {block_name}")

        # 3. 分割
        masks = segment_nuclei_by_method(imgs["DAPI"], method=self.seg_method)
        
        # 4. 获取胞质掩膜 (基于核向外扩张)
        cell_masks, cyto_only_masks = get_cytoplasm_masks(masks, expansion_distance=self.expansion_distance)

        # 5. 保存叠加图
        if self.save_overlay:
            out_tif = self.overlay_dir / f"{block_name}_nuclei_overlay.tif"
            save_nuclei_overlay(imgs["DAPI"], masks, out_tif, cell_masks=cell_masks)

        # 6. 特征提取
        df = extract_features(block_name, masks, cyto_only_masks, imgs, cell_masks=cell_masks)
        
        # 7. 自动评分
        df = score_markers(df)

        if "global_cell_id" not in df.columns:
            df["global_cell_id"] = df["block"].astype(str) + "_" + df["cell_id"].astype(str)

        return df

def list_blocks():
    crop_root = config.crop_root
    if not crop_root.exists():
        raise FileNotFoundError(f"裁剪结果根目录不存在: {crop_root}")
    return sorted([p.name for p in crop_root.iterdir() if p.is_dir()])

def run_batch():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seg", default=config.get("SEGMENTATION.MODEL_TYPE", "cellpose"), help="cellpose|stardist|watershed")
    parser.add_argument("--out-tag", default="")
    parser.add_argument("--no-overlay", action="store_true")
    parser.add_argument("--no-align", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip already processed blocks")
    args = parser.parse_args()

    seg_method = str(args.seg).strip().lower()
    out_tag = str(args.out_tag).strip()
    suffix = f"_{out_tag}" if out_tag else ""

    out_dir = config.batch_output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / f"all_blocks_cell_features{suffix}.csv"
    log_csv = out_dir / f"batch_log{suffix}.csv"
    overlay_dir = out_dir / f"overlays{suffix}"
    state_csv = out_dir / f"batch_state{suffix}.csv"

    blocks = list_blocks()
    logger.info(f"Found {len(blocks)} blocks: {blocks}")
    
    # 断点续跑逻辑
    processed_blocks = set()
    if args.resume and state_csv.exists():
        state_df = pd.read_csv(state_csv)
        processed_blocks = set(state_df[state_df['status'] == 'OK']['block'].tolist())
        logger.info(f"Resuming: skipping {len(processed_blocks)} already processed blocks.")

    processor = BlockProcessor(
        seg_method=seg_method,
        do_align=not args.no_align,
        save_overlay=not args.no_overlay,
        overlay_dir=overlay_dir,
        expansion_distance=config.expansion_distance
    )

    all_dfs = []
    logs = []

    # 如果是续跑，且最终 CSV 已存在，先读取
    if args.resume and out_csv.exists():
        try:
            all_dfs.append(pd.read_csv(out_csv))
        except Exception as e:
            logger.error(f"Failed to read existing CSV for resume: {e}")

    for b in blocks:
        if b in processed_blocks:
            continue

        try:
            df_b = processor.process(b)
            all_dfs.append(df_b)

            logs.append({
                "block": b,
                "status": "OK",
                "n_cells": int(len(df_b)),
                "error": ""
            })
            logger.info(f"[OK] {b}: {len(df_b)} cells")

        except Exception as e:
            logs.append({
                "block": b,
                "status": "FAIL",
                "n_cells": 0,
                "error": repr(e)
            })
            logger.error(f"[FAIL] {b}: {e}")
            logger.error(traceback.format_exc())

    # 保存日志和状态
    new_logs_df = pd.DataFrame(logs)
    if args.resume and state_csv.exists():
        old_state_df = pd.read_csv(state_csv)
        final_state_df = pd.concat([old_state_df, new_logs_df]).drop_duplicates(subset=['block'], keep='last')
    else:
        final_state_df = new_logs_df

    final_state_df.to_csv(state_csv, index=False, encoding="utf-8-sig")
    final_state_df.to_csv(log_csv, index=False, encoding="utf-8-sig") # 保持兼容性

    if len(all_dfs) == 0:
        logger.warning("No blocks processed successfully, nothing to save.")
        return

    df_all = pd.concat(all_dfs, ignore_index=True)
    # 确保唯一性 (如果续跑时读取了旧的)
    df_all = df_all.drop_duplicates(subset=['global_cell_id'])

    front_cols = ["global_cell_id", "block", "cell_id"]
    cols = front_cols + [c for c in df_all.columns if c not in front_cols]
    df_all = df_all[cols]

    df_all.to_csv(out_csv, index=False, encoding="utf-8-sig")
    logger.info(f"Saved total CSV: {out_csv}")
    logger.info(f"Total cells: {len(df_all)}")
    logger.info(f"Overlays saved to: {overlay_dir}")

if __name__ == "__main__":
    run_batch()
