import pandas as pd

df = pd.read_csv("A4_cell_features.csv")

# 假设你有控制样本估计出的均值+3SD等，
# 这里先示意：用简单阈值（后面你可以换成真实数值）
ER_thresh  = df["ER_mean"].mean()  + 3 * df["ER_mean"].std()
PR_thresh  = df["PR_mean"].mean()  + 3 * df["PR_mean"].std()
HER2_thresh = df["HER2_mean"].mean()  # 或者按 HER2 scoring 规则
Ki67_thresh = df["Ki67_mean"].quantile(0.8)  # 比如前 20% 当高增殖

df["ER_label"]   = (df["ER_mean"]   > ER_thresh).astype(int)
df["PR_label"]   = (df["PR_mean"]   > PR_thresh).astype(int)
df["HER2_label"] = (df["HER2_mean"] > HER2_thresh).astype(int)
df["Ki67_label"] = (df["Ki67_mean"] > Ki67_thresh).astype(int)

df.to_csv("A4_cell_features_labeled.csv", index=False)
