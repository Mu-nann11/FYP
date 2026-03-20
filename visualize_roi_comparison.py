import pandas as pd
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from fiji_stitcher.config import load_config

def visualize_comparison():
    # 1. 加载配置
    config = load_config()
    
    # 2. 设置路径 (优先使用配置中的路径)
    # 在 Docker 容器内，这些路径通常是 /results/...
    crop_dir = Path(config.get("CROP_OUTPUT_DIR", "/results/crop"))
    feature_dir = Path(config.get("FEATURE_OUTPUT_DIR", "/results/compared_result"))
    # QuPath 文件通常放在结果根目录
    results_root = crop_dir.parent 
    
    dapi_path = crop_dir / "A4" / "A4_DAPI_crop.tif"
    cp_csv = feature_dir / "A4_cell_features_roi_only.csv"
    qp_txt = results_root / "A4_DAPI_qupath_cells_manual.txt"
    out_img = feature_dir / "A4_ROI_Comparison.png"

    # 如果在本地运行且路径不存在，尝试使用硬编码的 Windows 路径作为备选
    if not dapi_path.exists():
        root = Path(r"D:\15_3_FYP_Munan\Code")
        dapi_path = root / "results" / "crop" / "A4" / "A4_DAPI_crop.tif"
        cp_csv = root / "results" / "compared_result" / "A4_cell_features_roi_only.csv"
        qp_txt = root / "results" / "A4_DAPI_qupath_cells_manual.txt"
        out_img = root / "results" / "compared_result" / "A4_ROI_Comparison.png"

    if not dapi_path.exists():
        print(f"ERROR: File not found: {dapi_path}")
        return

    # 3. 定义 ROI (从 filter_cellpose_by_roi.py 获取)
    cx_center = 1272.6803
    cy_center = 2081.2466
    width = 725.1
    height = 346.2

    xmin = cx_center - width / 2
    xmax = cx_center + width / 2
    ymin = cy_center - height / 2
    ymax = cy_center + height / 2

    print(f"ROI boundaries: x={xmin:.1f}-{xmax:.1f}, y={ymin:.1f}-{ymax:.1f}")

    # 4. 加载图像并进行 8-bit 归一化以便显示
    dapi = tifffile.imread(str(dapi_path))
    
    # 归一化到 0-255 (8-bit)
    dapi_float = dapi.astype(np.float32)
    dapi_min, dapi_max = dapi_float.min(), dapi_float.max()
    dapi_norm = ((dapi_float - dapi_min) / (dapi_max - dapi_min) * 255).astype(np.uint8)
    
    # 应用 CLAHE 增强对比度
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    dapi_norm = clahe.apply(dapi_norm)

    # 4. 裁剪图像到 ROI 区域
    # 注意: 数组索引是 [y, x]
    roi_img = dapi_norm[int(ymin):int(ymax), int(xmin):int(xmax)]

    # 5. 加载坐标数据
    df_cp = pd.read_csv(cp_csv)
    df_qp = pd.read_csv(qp_txt, sep="\t")

    # 过滤 QuPath 数据到 ROI 内 (Cellpose 数据已经是过滤过的)
    df_qp_roi = df_qp[
        (df_qp["Centroid X µm"] >= xmin) & (df_qp["Centroid X µm"] <= xmax) &
        (df_qp["Centroid Y µm"] >= ymin) & (df_qp["Centroid Y µm"] <= ymax)
    ].copy()

    # 6. 绘图
    plt.figure(figsize=(15, 8))
    
    # 显示 DAPI 图像
    plt.imshow(roi_img, cmap='gray', extent=[xmin, xmax, ymax, ymin])
    
    # 绘制 Cellpose 质心 (绿色圆圈)
    plt.scatter(df_cp["centroid_x"], df_cp["centroid_y"], 
                s=30, edgecolors='lime', facecolors='none', label='Cellpose (Auto)', alpha=0.8)
    
    # 绘制 QuPath 质心 (红色叉)
    plt.scatter(df_qp_roi["Centroid X µm"], df_qp_roi["Centroid Y µm"], 
                s=40, marker='x', color='red', label='QuPath (Manual)', alpha=0.8)

    plt.title(f"A4 ROI Segmentation Comparison\nCellpose: {len(df_cp)} cells | QuPath: {len(df_qp_roi)} cells")
    plt.legend()
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    
    # 反转 Y 轴以匹配图像坐标
    plt.gca().invert_yaxis()

    # 保存结果
    out_img.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_img, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {out_img}")

if __name__ == "__main__":
    visualize_comparison()
