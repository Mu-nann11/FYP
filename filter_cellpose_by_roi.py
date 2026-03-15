# filter_cellpose_by_roi.py

import pandas as pd
from pathlib import Path

# 1) 读取 Cellpose 特征表（Compared_Result 里的 CSV）
cp_path = Path(r"D:\FYP_TRY\20_12_FYP_1\Result\Compared_Result\A4_cell_features_with_labels.csv")
df_cp = pd.read_csv(cp_path)

# 质心列
cx_col = "centroid_x"
cy_col = "centroid_y"

print("列名：")
print(df_cp.columns.tolist())

# 2) 根据 QuPath 中矩形的中心点 + 长度定义 ROI
# 把下面两个数值改成你在 QuPath 信息栏看到的 Centroid X/µm 和 Centroid Y/µm
cx_center = 1272.6803
cy_center = 2081.2466

# 你给出的 height = 346.2，就用正方形 ROI（宽 = 高 = 346.2）
width = 725.1
height = 346.2

xmin = cx_center - width / 2
xmax = cx_center + width / 2
ymin = cy_center - height / 2
ymax = cy_center + height / 2

print("\nROI 边界：")
print("xmin, xmax =", xmin, xmax)
print("ymin, ymax =", ymin, ymax)

# 3) 按 ROI 过滤细胞
df_roi = df_cp[
    (df_cp[cx_col] >= xmin) & (df_cp[cx_col] <= xmax) &
    (df_cp[cy_col] >= ymin) & (df_cp[cy_col] <= ymax)
].copy()

# 4) 打印统计信息
print("\n=== Cellpose 计数 ===")
print("全图细胞数:", len(df_cp))
print("ROI 内细胞数:", len(df_roi))

print("\n=== ROI 内面积统计 ===")
print("area mean/median:",
      df_roi["area"].mean(), df_roi["area"].median())

# 5) 保存 ROI 内细胞到新文件（固定目录 Compared_Result）
out_dir = Path(r"D:\FYP_TRY\20_12_FYP_1\Result\Compared_Result")
out_dir.mkdir(parents=True, exist_ok=True)

out_path = out_dir / "A4_cell_features_roi_only.csv"
df_roi.to_csv(out_path, index=False, encoding="utf-8-sig")
print("\n已保存 ROI 内细胞到:", out_path)
