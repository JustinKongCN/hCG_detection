import os
import random
import logging
import numpy as np
import torch

def set_seed(seed=42):
    """固定随机种子，保证每次实验结果一样"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def setup_logger(log_dir="./logs"):
    """创建日志，自动写文件"""
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(f"{log_dir}/train.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def ensure_dirs(*dirs):
    """自动创建不存在的文件夹"""
    for d in dirs:
        os.makedirs(d, exist_ok=True)