import numpy as np
import logging
import sys
from pathlib import Path

def get_logger(name: str, log_file: Path = None, level=logging.INFO):
    """
    统一日志记录器
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # 控制台输出
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # 文件输出
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
    return logger

def normalize_to_uint16(img: np.ndarray) -> np.ndarray:
    """
    将图像归一化到 uint16 范围 (0-65535)
    """
    img = img.astype(np.float32)
    img -= img.min()
    if img.max() > 0:
        img /= img.max()
    img = (img * 65535.0).clip(0, 65535)
    return img.astype(np.uint16)

def q90(arr: np.ndarray) -> float:
    """
    计算 90% 分位数
    """
    if arr.size == 0:
        return float("nan")
    return float(np.quantile(arr, 0.90))
