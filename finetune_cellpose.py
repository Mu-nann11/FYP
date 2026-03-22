"""
Cellpose Fine-tuning Script

用法:
  # 从预训练 nuclei 模型微调
  python finetune_cellpose.py --data-dir /data/training --base-model nuclei --output /models/finetuned_nuclei

  # 继续微调已有模型
  python finetune_cellpose.py --data-dir /data/training --base-model /models/finetuned_nuclei --output /models/finetuned_v2

  # 使用 GPU / CPU
  python finetune_cellpose.py --data-dir /data/training --gpu
  python finetune_cellpose.py --data-dir /data/training --no-gpu

训练数据目录结构 (Cellpose 标准格式):
  data_dir/
    sample01_img.tif        # 灰度或 RGB 图像
    sample01_masks.tif      # 对应的 label mask (int32, 0=背景, 1..N=细胞)
    sample02_img.tif
    sample02_masks.tif
    ...

  也支持 .npy 格式:
    sample01_img.npy
    sample01_masks.npy

  或者 images/ + masks/ 子目录:
    data_dir/images/sample01.tif
    data_dir/masks/sample01_masks.tif
"""

import argparse
import os
import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np


def discover_training_pairs(data_dir: Path):
    """
    扫描目录，发现 (image, mask) 训练对。
    支持:
      - 同目录: xxx_img.tif + xxx_masks.tif
      - 同目录: xxx_img.npy + xxx_masks.npy
      - 子目录: images/xxx.tif + masks/xxx_masks.tif
    返回: list of (image_path, mask_path)
    """
    pairs = []
    data_dir = Path(data_dir)

    # 模式 1: images/ + masks/ 子目录
    img_subdir = data_dir / "images"
    mask_subdir = data_dir / "masks"
    if img_subdir.is_dir() and mask_subdir.is_dir():
        for img_file in sorted(img_subdir.iterdir()):
            if img_file.suffix not in (".tif", ".tiff", ".npy", ".png"):
                continue
            stem = img_file.stem.replace("_img", "")
            # 尝试多种 mask 命名
            for mask_name in [f"{stem}_masks.tif", f"{stem}_masks.tiff",
                              f"{stem}_masks.npy", f"{stem}_seg.npy"]:
                mask_file = mask_subdir / mask_name
                if mask_file.exists():
                    pairs.append((img_file, mask_file))
                    break
            else:
                # 精确匹配
                mask_file = mask_subdir / f"{img_file.stem}_masks{img_file.suffix}"
                if mask_file.exists():
                    pairs.append((img_file, mask_file))

        if pairs:
            print(f"  子目录模式: 发现 {len(pairs)} 对训练数据")
            return pairs

    # 模式 2: 同目录 _img + _masks
    all_files = sorted(data_dir.iterdir())
    img_candidates = {}
    for f in all_files:
        if f.suffix not in (".tif", ".tiff", ".npy", ".png"):
            continue
        name = f.stem
        if name.endswith("_img"):
            base = name[:-4]  # 去掉 _img
            img_candidates[base] = f

    for base, img_file in img_candidates.items():
        for mask_suffix in ["_masks", "_seg"]:
            for ext in [".tif", ".tiff", ".npy"]:
                mask_file = data_dir / f"{base}{mask_suffix}{ext}"
                if mask_file.exists():
                    pairs.append((img_file, mask_file))
                    break
            else:
                continue
            break

    if not pairs:
        # 模式 3: 没有 _img 后缀，尝试按文件名匹配
        all_imgs = [f for f in all_files if f.suffix in (".tif", ".tiff", ".png", ".npy")
                    and "_masks" not in f.stem and "_seg" not in f.stem]
        for img_file in sorted(all_imgs):
            for mask_suffix in ["_masks", "_seg"]:
                for ext in [".tif", ".tiff", ".npy"]:
                    mask_file = data_dir / f"{img_file.stem}{mask_suffix}{ext}"
                    if mask_file.exists():
                        pairs.append((img_file, mask_file))
                        break
                else:
                    continue
                break

    print(f"  共发现 {len(pairs)} 对训练数据")
    return pairs


def load_training_data(pairs):
    """加载所有训练图像和 mask。"""
    import tifffile

    images, masks = [], []
    for img_path, mask_path in pairs:
        # 加载图像
        if str(img_path).endswith(".npy"):
            img = np.load(str(img_path))
        else:
            img = tifffile.imread(str(img_path))

        # 加载 mask
        if str(mask_path).endswith(".npy"):
            msk = np.load(str(mask_path))
        else:
            msk = tifffile.imread(str(mask_path))

        # 确保 mask 是 int32
        msk = msk.astype(np.int32)

        # 如果 mask 有多个 frame，取第一个
        if msk.ndim == 3:
            msk = msk[0]
        if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
            img = img[0] if img.shape[0] == 1 else img

        images.append(img)
        masks.append(msk)

    return images, masks


def run_finetune(
    data_dir: str,
    base_model: str = "nuclei",
    output_path: str = "/models/finetuned_cellpose",
    use_gpu: bool = None,
    n_epochs: int = 100,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-4,
    batch_size: int = 8,
    channel_axis: int = None,
    normalize: bool = True,
    min_train_masks: int = 5,
    model_type: str = "cyto3",
):
    """
    执行 Cellpose fine-tuning。

    参数:
        data_dir: 训练数据目录
        base_model: 基础模型 ("nuclei", "cyto", "cyto2", "cyto3", 或自定义模型路径)
        output_path: 微调模型保存路径
        use_gpu: 是否使用 GPU
        n_epochs: 训练轮数
        learning_rate: 学习率
        weight_decay: 权重衰减
        batch_size: 批次大小
        channel_axis: 通道轴索引（None=自动检测）
        normalize: 是否自动归一化
        min_train_masks: 最少训练 mask 数
        model_type: Cellpose 模型类型 (cyto3 / cyto2 / nuclei)
    """
    from cellpose import models, train

    data_dir = Path(data_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if use_gpu is None:
        import torch
        use_gpu = torch.cuda.is_available()

    print(f"=" * 60)
    print(f"Cellpose Fine-tuning")
    print(f"=" * 60)
    print(f"  数据目录:   {data_dir}")
    print(f"  基础模型:   {base_model}")
    print(f"  输出路径:   {output_path}")
    print(f"  GPU:        {use_gpu}")
    print(f"  Epochs:     {n_epochs}")
    print(f"  LR:         {learning_rate}")
    print(f"  Batch size: {batch_size}")
    print(f"=" * 60)

    # 1. 发现训练数据
    print("\n[1/4] 扫描训练数据...")
    pairs = discover_training_pairs(data_dir)
    if not pairs:
        print("❌ 未发现训练数据！请检查目录结构。")
        sys.exit(1)

    # 2. 加载数据
    print("[2/4] 加载训练数据...")
    images, masks = load_training_data(pairs)
    print(f"  已加载 {len(images)} 张图像")

    # 过滤掉 mask 数量不足的样本
    valid = [(img, msk) for img, msk in zip(images, masks) if int(msk.max()) >= min_train_masks]
    if len(valid) < len(images):
        print(f"  过滤掉 {len(images) - len(valid)} 张 mask 数 < {min_train_masks} 的样本")
    if not valid:
        print(f"❌ 所有样本的 mask 数都 < {min_train_masks}，无法训练！")
        sys.exit(1)
    images, masks = zip(*valid)
    images, masks = list(images), list(masks)

    # 3. 初始化模型
    print("[3/4] 初始化模型...")

    # 判断 base_model 是预训练名称还是路径
    pretrained_models = {"nuclei", "cyto", "cyto2", "cyto3", "cpsam", "tissuenet", "livecell"}
    if base_model in pretrained_models:
        print(f"  从预训练模型 '{base_model}' 微调")
        model = models.CellposeModel(gpu=use_gpu, model_type=base_model)
    elif os.path.isfile(base_model):
        print(f"  从自定义模型加载: {base_model}")
        model = models.CellposeModel(gpu=use_gpu, pretrained_model=base_model)
    else:
        print(f"  ⚠️ 基础模型 '{base_model}' 未找到，尝试作为预训练名称加载")
        model = models.CellposeModel(gpu=use_gpu, model_type=base_model)

    # 4. 训练
    print(f"[4/4] 开始训练 ({n_epochs} epochs)...")
    model_path = str(output_path).replace(".pth", "")  # cellpose 自动加后缀

    # Cellpose 4.x: model.train() 接口
    try:
        train_data, train_labels, train_files = model._load_train_data(
            train_dir=str(data_dir),
            mask_filter="_masks",
        )
    except (AttributeError, TypeError):
        # 直接用已加载的数据
        train_data = images
        train_labels = masks
        train_files = [str(p[0]) for p in pairs]

    # 使用 model.train() 进行 fine-tune
    # Cellpose 2+/3+ 的 train 方法
    try:
        new_model_path = train.train_seg(
            model.net,
            train_data=train_data,
            train_labels=train_labels,
            train_files=train_files,
            test_data=None,
            test_labels=None,
            test_files=None,
            normalize=normalize,
            channels=[channel_axis, 0] if channel_axis is not None else None,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            model_name=output_path.stem,
            save_path=str(output_path.parent),
        )
        print(f"\n✅ 模型已保存: {new_model_path}")
    except (AttributeError, TypeError) as e:
        # 旧版 cellpose 接口 fallback
        print(f"  train_seg 接口不兼容 ({e})，尝试旧版 model.train()...")
        try:
            model.train(
                train_data=train_data,
                train_labels=train_labels,
                channels=[channel_axis, 0] if channel_axis is not None else None,
                normalize=normalize,
                save_path=str(output_path.parent),
                n_epochs=n_epochs,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                batch_size=batch_size,
            )
            print(f"\n✅ 模型已保存到: {output_path.parent}")
        except Exception as e2:
            print(f"\n❌ 训练失败: {e2}")
            print("请检查 Cellpose 版本和数据格式。")
            sys.exit(1)

    # 5. 保存训练元信息
    meta = {
        "timestamp": datetime.now().isoformat(),
        "base_model": base_model,
        "output_path": str(output_path),
        "n_samples": len(images),
        "n_epochs": n_epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "use_gpu": use_gpu,
        "data_dir": str(data_dir),
    }
    meta_path = output_path.parent / f"{output_path.stem}_training_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  训练元信息: {meta_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Cellpose Fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--data-dir", required=True, help="训练数据目录")
    parser.add_argument("--base-model", default="nuclei",
                        help="基础模型: nuclei/cyto/cyto2/cyto3 或自定义 .pth 路径")
    parser.add_argument("--output", default="/models/finetuned_cellpose",
                        help="微调模型输出路径 (不含后缀)")
    parser.add_argument("--gpu", dest="use_gpu", action="store_true", default=None,
                        help="强制使用 GPU")
    parser.add_argument("--no-gpu", dest="use_gpu", action="store_false",
                        help="强制使用 CPU")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-5, help="学习率")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="权重衰减")
    parser.add_argument("--batch-size", type=int, default=8, help="批次大小")
    parser.add_argument("--channel-axis", type=int, default=None,
                        help="通道轴索引 (None=自动检测)")
    parser.add_argument("--no-normalize", action="store_true",
                        help="禁用自动归一化")
    parser.add_argument("--min-masks", type=int, default=5,
                        help="过滤少于 N 个 mask 的训练样本")

    args = parser.parse_args()

    run_finetune(
        data_dir=args.data_dir,
        base_model=args.base_model,
        output_path=args.output,
        use_gpu=args.use_gpu,
        n_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        channel_axis=args.channel_axis,
        normalize=not args.no_normalize,
        min_train_masks=args.min_masks,
    )


if __name__ == "__main__":
    main()
