import torch
import torch.nn as nn
from tqdm import tqdm

class Trainer:
    def __init__(self, model, dataset, config, device, logger):
        self.model = model.to(device)
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
        self.cfg = config
        self.device = device
        self.logger = logger
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
        self.criterion_reg = nn.MSELoss()
        self.criterion_cls = nn.CrossEntropyLoss()

    def run(self):
        self.model.train()
        for epoch in range(self.cfg["epochs"]):
            total_loss = 0
            for batch in tqdm(self.dataloader, desc=f"Epoch {epoch+1}"):
                batch = batch.to(self.device)
                # 注意：这里batch暂时没标签，你需要后续补上浓度和风险标签
                # 现在先跑通结构
                conc, risk = self.model(batch)
                # 假标签，仅演示结构
                loss = conc.mean() + risk.mean()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            self.logger.info(f"Epoch {epoch+1}/{self.cfg['epochs']}, Loss: {total_loss:.4f}")
            # 保存模型
            torch.save(self.model.state_dict(), f"{self.cfg['checkpoint_dir']}/epoch_{epoch+1}.pth")