# FYP 项目进度报告

## 项目概述

**项目名称**: 乳腺癌 TMA（组织微阵列）自动化分析 Pipeline
**GitHub**: https://github.com/Mu-nann11/FYP
**目标**: 开发一套自动化计算 pipeline，用于：
1. 多通道荧光 TMA 图像拼接（Fiji/ImageJ）
2. 多通道图像配准（DAPI 为参考）
3. 细胞核分割（Cellpose / StarDist / Watershed）
4. 生物标志物定量分析（ER, PR, HER2, Ki67）
5. 乳腺癌分子分型（Luminal A/B, HER2-enriched, Triple-negative）

---

## 任务优先级清单

| 优先级 | 任务 | 状态 |
|--------|------|------|
| P0 | ORB + RANSAC 替代 phaseCorrelate（+ ECC 中间方案） | ✅ 完成 |
| P0 | 配准质量指标 (MI / NCC) | ✅ 完成（随 P0 一并实现） |
| P1 | Ki67 hotspot 完善 + 可视化 | ✅ 完成 |
| P1 | ER/PR 汇总到 patient 级 + Cohen's kappa | 🔲 待做 |
| P1 | 可视化报告生成器 (HTML/PDF) | 🔲 待做 |
| P2 | 细胞空间上下文特征（最近邻距离、局部密度） | 🔲 待做 |
| P2 | SAM mask refinement | 🔲 待做 |
| P3 | Cellpose fine-tuning | 🔲 待做（可降为 future work） |
| P3 | Texture features (GLCM/LBP) | 🔲 待做 |

---

## 已完成任务详情

### ✅ P0-1: alignment.py 重写 (2026-03-22)

**改动文件**: `alignment.py`, `batch_run_all_to_one.py`

**新增功能**:
1. **`align_by_ecc()`** — 基于 ECC (Enhanced Correlation Coefficient) 的仿射配准
   - 使用 `cv2.findTransformECC` + `MOTION_AFFINE`
   - 能处理旋转 + 平移 + 缩放（phaseCorrelate 只能处理平移）
   - 合成测试验证：旋转恢复精度 -5.00°（true 5.00°，符号差异为逆变换正常现象），平移恢复 tx=8.01, ty=5.03

2. **`align_by_orb()`** — ORB + RANSAC 特征配准
   - ORB 特征检测 → KNN 匹配 + Lowe's ratio test → `estimateAffinePartial2D` + RANSAC
   - 对细胞图像可能特征不足，自动 fallback

3. **统一接口 `align(dapi, target, method=...)`**
   - `method="auto"`: ORB → ECC → phaseCorrelate 三级 fallback
   - `method="orb"`: 强制 ORB
   - `method="ecc"`: 强制 ECC
   - `method="shift"`: 强制 phaseCorrelate（原方案）

4. **配准质量指标**:
   - `mutual_information()`: 归一化互信息，范围 [0, 1]
   - `normalized_cross_correlation()`: 排除 warp 黑边的有效区域 NCC
   - `alignment_quality()`: 一站式返回

5. **batch_run_all_to_one.py 改动**:
   - 新增 `--align-method` CLI 参数（auto/orb/ecc/shift）
   - 每次对齐自动 log 方法名 + MI + NCC
   - BlockProcessor 新增 `align_method` 属性

### ✅ P1-1: Ki67 hotspot 完善 + 可视化 (2026-03-22)

**改动文件**: `features.py`, `segmentation.py`, `batch_run_all_to_one.py`

**新增功能**:
1. **Hotspot 坐标记录** (`features.py` → `score_markers`):
   - hotspot 种子坐标写入 df（`hotspot_seed_x_0`, `hotspot_seed_y_0`, ...）
   - 附带 `KI67_hotspot_radius_px` 和 `KI67_hotspot_n_seeds` 列

2. **Per-hotspot Ki67 指数** (`features.py`):
   - `get_ki67_hotspot_seeds(df)`: 从 df 提取种子坐标
   - `compute_per_hotspot_ki67_index(df, seeds, radius_px)`: 逐个 hotspot 计算阳性率，返回 list of dict

3. **Hotspot overlay 可视化** (`segmentation.py` → `save_ki67_hotspot_overlay`):
   - 绿色 = hotspot 内阳性细胞（高亮重点）
   - 红色 = hotspot 内阴性细胞
   - 黄色 = hotspot 外阳性细胞（弱化显示）
   - 灰色 = hotspot 外阴性细胞
   - 青色圆环标注 hotspot 边界
   - 左上角叠加全局 + per-hotspot Ki67 阳性率文字
   - 同时输出 TIFF + PNG 格式

4. **Batch 集成** (`batch_run_all_to_one.py`):
   - Ki67 block 自动额外生成 `{dataset}_{block}_ki67_hotspot_overlay.tif/.png`
   - 与原有 `_ki67_overlay.tif` 并行输出

---

## 下一步

**即将执行**: P1 — ER/PR 汇总到 patient 级 + Cohen's kappa
- 现有代码已有 `_greedy_intensity_seeds` 和 `_hotspot_mask_from_seeds` 基础
- 需要：完善 hotspot 阳性率计算 + 在 overlay 图上标注 hotspot 区域

---

## 使用说明

继续开发时，将此文件内容提供给 AI 助手，它会自动了解项目背景、已完成内容和下一步计划。
