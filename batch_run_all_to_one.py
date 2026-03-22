import argparse
import traceback
from pathlib import Path
from typing import Optional, List

import pandas as pd

from loader import load_block, DATASETS
from alignment import align
from segmentation import segment_nuclei_by_method, get_cytoplasm_masks, save_nuclei_overlay, save_ki67_overlay, save_ki67_hotspot_overlay, refine_masks_with_sam
from features import extract_features, score_markers, compute_ki67_index, compute_ki67_hotspot_index, get_ki67_hotspot_seeds, compute_per_hotspot_ki67_index
from patient_aggregation import aggregate_to_patient, block_level_kappa, patient_level_kappa
from report_generator import generate_report
from utils import get_logger
from config import config

# 获取统一日志记录器
logger = get_logger("batch_run", log_file=config.batch_output_dir / "batch_run.log")

class BlockProcessor:
    def __init__(
        self,
        dataset: str = "TMAe",
        seg_method: str = "cellpose",
        do_align: bool = True,
        align_method: str = "auto",
        save_overlay: bool = True,
        overlay_dir: Optional[Path] = None,
        expansion_distance: int = 15,
        sam_refine: bool = False,
    ):
        self.dataset = dataset
        self.seg_method = seg_method
        self.do_align = do_align
        self.align_method = align_method
        self.save_overlay = save_overlay
        self.overlay_dir = overlay_dir or (config.batch_output_dir / "overlays")
        self.expansion_distance = expansion_distance
        self.sam_refine = sam_refine

        if self.save_overlay:
            self.overlay_dir.mkdir(parents=True, exist_ok=True)

    def process(self, block_name: str) -> pd.DataFrame:
        logger.info(f"Processing block: {self.dataset}/{block_name}")

        # 1. 加载数据
        data = load_block(self.dataset, block_name, do_preprocess=True)

        # 2. 提取通道 (TMAe: cycle1; TMAd: cycle1 + cycle2)
        cycle1 = data["cycle1"]
        dapi = cycle1["DAPI"]

        # 构建传给 extract_features 的通道字典，key 前缀区分 cycle
        channels_dict = {ch: cycle1[ch] for ch in cycle1 if ch != "DAPI"}

        if self.do_align:
            for ch in list(channels_dict.keys()):
                aligned, _, info = align(dapi, channels_dict[ch], method=self.align_method)
                channels_dict[ch] = aligned
                logger.info(f"  Alignment [{ch}]: method={info.get('method')}, "
                            f"MI={info.get('mutual_information', 'N/A'):.3f}, "
                            f"NCC={info.get('ncc', 'N/A'):.3f}")

        # TMAd: 加入 cycle2 的 KI67 通道
        ki67_img = None
        if "cycle2" in data:
            cycle2 = data["cycle2"]
            # cycle2 的 DAPI 用于对齐 KI67，但分割仍用 cycle1 DAPI
            if "KI67" in cycle2:
                ki67_img = cycle2["KI67"]
                if self.do_align and "DAPI" in cycle2:
                    ki67_img, _, info = align(dapi, ki67_img, method=self.align_method)
                    logger.info(f"  Alignment [KI67]: method={info.get('method')}, "
                                f"MI={info.get('mutual_information', 'N/A'):.3f}, "
                                f"NCC={info.get('ncc', 'N/A'):.3f}")
                channels_dict["KI67"] = ki67_img

        # 3. 分割 (始终用 cycle1 DAPI)
        masks = segment_nuclei_by_method(dapi, method=self.seg_method)

        # 3b. SAM 精炼 (可选)
        if self.sam_refine:
            sam_min_area = int(config.get("SEGMENTATION.SAM_MIN_AREA", 50))
            sam_box_pad = int(config.get("SEGMENTATION.SAM_BOX_PAD", 3))
            logger.info(f"  SAM refinement: min_area={sam_min_area}, box_pad={sam_box_pad}")
            try:
                masks = refine_masks_with_sam(dapi, masks, min_area=sam_min_area, box_pad=sam_box_pad)
                logger.info(f"  SAM refinement done: {int(masks.max())} labels")
            except (ImportError, FileNotFoundError) as e:
                logger.warning(f"  SAM refinement skipped: {e}")
                print(f"⚠️ SAM refinement skipped: {e}")

        # 4. 获取胞质掩膜
        cell_masks, cyto_only_masks = get_cytoplasm_masks(masks, expansion_distance=self.expansion_distance)

        # 5. 保存叠加图
        if self.save_overlay:
            out_tif = self.overlay_dir / f"{self.dataset}_{block_name}_nuclei_overlay.tif"
            save_nuclei_overlay(dapi, masks, out_tif, cell_masks=cell_masks)

            if ki67_img is not None:
                ki67_tif = self.overlay_dir / f"{self.dataset}_{block_name}_ki67_overlay.tif"
                save_ki67_overlay(dapi, masks, ki67_img, ki67_tif)

        # 6. 特征提取
        df = extract_features(block_name, masks, cyto_only_masks, channels_dict, cell_masks=cell_masks)

        # 7. 自动评分
        df = score_markers(df)

        # 8. Ki67 增殖指数 (仅 TMAd)
        if ki67_img is not None and not df.empty:
            df["ki67_proliferation_index"] = compute_ki67_index(df)
            df["ki67_hotspot_proliferation_index"] = compute_ki67_hotspot_index(df)

            # hotspot overlay 可视化
            if self.save_overlay:
                seeds = get_ki67_hotspot_seeds(df)
                radius_px = float(config.get("SCORING.KI67_HOTSPOT_RADIUS_PX", 150.0))
                per_hotspot = compute_per_hotspot_ki67_index(df, seeds, radius_px) if seeds else None
                hotspot_tif = self.overlay_dir / f"{self.dataset}_{block_name}_ki67_hotspot_overlay.tif"
                save_ki67_hotspot_overlay(
                    dapi, masks, ki67_img, df,
                    seeds=seeds, radius_px=radius_px,
                    out_tif_path=hotspot_tif,
                    per_hotspot_stats=per_hotspot,
                )
                logger.info(f"  Ki67 hotspot overlay saved: {hotspot_tif}")

        # global_cell_id 加入 dataset 前缀，避免 TMAe/TMAd 同名 block 冲突
        df["dataset"] = self.dataset
        df["global_cell_id"] = self.dataset + "_" + df["block"].astype(str) + "_" + df["cell_id"].astype(str)

        return df


def list_blocks(dataset: str) -> List[str]:
    crop_root = config.crop_root / dataset
    if not crop_root.exists():
        raise FileNotFoundError(f"裁剪结果根目录不存在: {crop_root}")
    return sorted([p.name for p in crop_root.iterdir() if p.is_dir()])


def run_batch():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="TMAe", choices=list(DATASETS.keys()), help="数据集名称")
    parser.add_argument("--seg", default=config.get("SEGMENTATION.MODEL_TYPE", "cellpose"), help="cellpose|stardist|watershed")
    parser.add_argument("--out-tag", default="")
    parser.add_argument("--no-overlay", action="store_true")
    parser.add_argument("--no-align", action="store_true")
    parser.add_argument("--align-method", default="auto", choices=["auto", "orb", "ecc", "shift"],
                        help="Alignment method: auto (ORB→ECC→phase corr), orb, ecc, shift")
    parser.add_argument("--resume", action="store_true", help="Skip already processed blocks")
    parser.add_argument("--sam-refine", action="store_true", help="Use SAM to refine segmentation masks")
    parser.add_argument("--qupath-csv", default=None,
                        help="Path to QuPath comparison CSV (from compare_qupath) for Cohen's kappa")
    parser.add_argument("--no-report", action="store_true",
                        help="Skip HTML report generation")
    args = parser.parse_args()

    dataset = args.dataset
    seg_method = str(args.seg).strip().lower()
    out_tag = str(args.out_tag).strip()
    suffix = f"_{dataset}_{out_tag}" if out_tag else f"_{dataset}"

    out_dir = config.batch_output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / f"all_blocks_cell_features{suffix}.csv"
    patient_csv = out_dir / f"patient_level_scores{suffix}.csv"
    log_csv = out_dir / f"batch_log{suffix}.csv"
    overlay_dir = out_dir / f"overlays{suffix}"
    state_csv = out_dir / f"batch_state{suffix}.csv"

    blocks = list_blocks(dataset)
    logger.info(f"Dataset: {dataset}, Found {len(blocks)} blocks: {blocks}")

    # 断点续跑逻辑
    processed_blocks = set()
    if args.resume and state_csv.exists():
        state_df = pd.read_csv(state_csv)
        processed_blocks = set(state_df[state_df['status'] == 'OK']['block'].tolist())
        logger.info(f"Resuming: skipping {len(processed_blocks)} already processed blocks.")

    processor = BlockProcessor(
        dataset=dataset,
        seg_method=seg_method,
        do_align=not args.no_align,
        align_method=args.align_method,
        save_overlay=not args.no_overlay,
        overlay_dir=overlay_dir,
        expansion_distance=config.expansion_distance,
        sam_refine=args.sam_refine or bool(config.get("SEGMENTATION.SAM_REFINE", False)),
    )

    all_dfs = []
    logs = []

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
            logger.info(f"[OK] {dataset}/{b}: {len(df_b)} cells")

        except Exception as e:
            logs.append({
                "block": b,
                "status": "FAIL",
                "n_cells": 0,
                "error": repr(e)
            })
            logger.error(f"[FAIL] {dataset}/{b}: {e}")
            logger.error(traceback.format_exc())

    new_logs_df = pd.DataFrame(logs)
    if args.resume and state_csv.exists():
        old_state_df = pd.read_csv(state_csv)
        final_state_df = pd.concat([old_state_df, new_logs_df]).drop_duplicates(subset=['block'], keep='last')
    else:
        final_state_df = new_logs_df

    final_state_df.to_csv(state_csv, index=False, encoding="utf-8-sig")
    final_state_df.to_csv(log_csv, index=False, encoding="utf-8-sig")

    if len(all_dfs) == 0:
        logger.warning("No blocks processed successfully, nothing to save.")
        return

    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all = df_all.drop_duplicates(subset=['global_cell_id'])

    front_cols = ["global_cell_id", "dataset", "block", "cell_id"]
    cols = front_cols + [c for c in df_all.columns if c not in front_cols]
    df_all = df_all[cols]

    df_all.to_csv(out_csv, index=False, encoding="utf-8-sig")
    logger.info(f"Saved total CSV: {out_csv}")
    logger.info(f"Total cells: {len(df_all)}")
    logger.info(f"Overlays saved to: {overlay_dir}")

    # =====================================================
    # Patient-level ER/PR aggregation
    # =====================================================
    logger.info("Aggregating ER/PR scores to patient level...")
    min_pos_frac = float(config.get("SCORING.ER_PR_MIN_POSITIVE_FRACTION", 0.01))
    patient_df = aggregate_to_patient(df_all, min_pos_fraction=min_pos_frac)
    patient_df.to_csv(patient_csv, index=False, encoding="utf-8-sig")
    logger.info(f"Saved patient-level scores: {patient_csv}")
    logger.info(f"Patients: {patient_df['patient_id'].nunique()}")
    for _, row in patient_df.iterrows():
        logger.info(f"  {row['patient_id']} / {row['marker']}: "
                     f"{row['patient_status']} "
                     f"(positive fraction={row['positive_fraction']:.4f}, "
                     f"n={row['total_cells']}, blocks={row['n_blocks']})")

    # =====================================================
    # Cohen's kappa vs QuPath (if reference provided)
    # =====================================================
    if args.qupath_csv:
        logger.info(f"Computing Cohen's kappa vs QuPath: {args.qupath_csv}")
        try:
            qupath_df = pd.read_csv(args.qupath_csv, encoding="utf-8-sig")

            # Block-level (cell-matched) kappa
            kappa_block = block_level_kappa(df_all, qupath_df)
            if not kappa_block.empty:
                kappa_block_path = out_dir / f"kappa_cell_level{suffix}.csv"
                kappa_block.to_csv(kappa_block_path, index=False, encoding="utf-8-sig")
                logger.info(f"Cell-level kappa saved: {kappa_block_path}")
                for _, row in kappa_block.iterrows():
                    logger.info(f"  {row['marker']}: kappa={row.get('kappa', 'N/A')}, "
                                f"agreement={row.get('agreement_rate', 'N/A')}, "
                                f"n={row.get('n_samples', 0)}")

            # Patient-level kappa (if QuPath CSV also has block/patient info)
            if "patient_id" in qupath_df.columns and "marker" in qupath_df.columns and "patient_status" in qupath_df.columns:
                kappa_patient = patient_level_kappa(patient_df, qupath_df)
                kappa_patient_path = out_dir / f"kappa_patient_level{suffix}.csv"
                kappa_patient.to_csv(kappa_patient_path, index=False, encoding="utf-8-sig")
                logger.info(f"Patient-level kappa saved: {kappa_patient_path}")
                for _, row in kappa_patient.iterrows():
                    logger.info(f"  {row['marker']}: kappa={row['kappa']}, "
                                f"agreement={row['agreement_rate']}, "
                                f"n_patients={row['n_patients']}")

        except Exception as e:
            logger.error(f"Kappa computation failed: {e}")
            logger.error(traceback.format_exc())
            print(f"⚠️ Kappa computation failed: {e}")

    # =====================================================
    # HTML Report
    # =====================================================
    kappa_csv_for_report = None
    kappa_cell_path = out_dir / f"kappa_cell_level{suffix}.csv"
    if kappa_cell_path.exists():
        kappa_csv_for_report = kappa_cell_path

    if not args.no_report:
        try:
            report_path = out_dir / f"report{suffix}.html"
            generate_report(
                cell_csv=out_csv,
                patient_csv=patient_csv if patient_csv.exists() else None,
                kappa_csv=kappa_csv_for_report,
                overlay_dir=overlay_dir if overlay_dir.exists() else None,
                out_html=report_path,
                title=f"TMA {dataset} Analysis Report",
            )
            logger.info(f"HTML report: {report_path}")
            print(f"📊 Report: {report_path}")
        except Exception as e:
            logger.error(f"Report generation failed: {e}")

    logger.info("Batch run complete.")

if __name__ == "__main__":
    run_batch()
