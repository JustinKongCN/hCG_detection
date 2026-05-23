import numpy as np
import argparse
import yaml
import torch
from pathlib import Path

from src.utils import setup_logger, set_seed, ensure_dirs
from src.extractor import HCG_FeatureExtractor
from src.dataset import HCGDataset
from src.models import hCG_TCN_MultiTask
from src.trainer import Trainer
from src.evaluator import Evaluator

def parse_args():
    parser = argparse.ArgumentParser(description="hCG智能POCT系统")
    parser.add_argument("--mode", choices=["extract", "train", "eval"], required=True, help="运行模式")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("--video", default=None, help="eval模式用的视频路径")
    return parser.parse_args()

def main():
    args = parse_args()

    # 1. 读配方本
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 2. 准备工作环境
    set_seed(42)
    logger = setup_logger("./logs")
    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")
    logger.info(f"===== 启动模式: {args.mode} | 设备: {device} =====")

    # 3. 按模式派活
    if args.mode == "extract":
        # 切配模式：视频 → .npy
        ensure_dirs(cfg["data"]["processed_dir"])
        extractor = HCG_FeatureExtractor(**cfg["data"])
        raw_dir = Path(cfg["data"]["raw_dir"])
        for video_file in raw_dir.glob("*.mp4"):
            mat = extractor.process(str(video_file))
            out_path = Path(cfg["data"]["processed_dir"]) / f"{video_file.stem}.npy"
            np.save(out_path, mat)
            logger.info(f"提取完成: {video_file.name} → {mat.shape}")

    elif args.mode == "train":
        # 训练模式：.npy → 模型
        ensure_dirs(cfg["train"]["checkpoint_dir"])
        dataset = HCGDataset(cfg["data"]["processed_dir"], cfg["data"]["seq_length"])
        model = hCG_TCN_MultiTask(**cfg["model"])
        trainer = Trainer(model, dataset, cfg["train"], device, logger)
        trainer.run()

    elif args.mode == "eval":
        # 推理模式：视频 → 直接出结果
        if not args.video:
            raise ValueError("eval模式需要 --video 参数指定视频路径")
        extractor = HCG_FeatureExtractor(**cfg["data"])
        model = hCG_TCN_MultiTask(**cfg["model"])
        # 加载训练好的权重（如果有）
        # model.load_state_dict(torch.load("checkpoints/epoch_100.pth"))
        evaluator = Evaluator(model, extractor, device, logger)
        result = evaluator.predict(args.video)
        logger.info(f"预测结果: {result}")

if __name__ == "__main__":
    main()