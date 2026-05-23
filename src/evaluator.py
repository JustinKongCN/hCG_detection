import torch
import numpy as np

class Evaluator:
    def __init__(self, model, extractor, device, logger):
        self.model = model.to(device).eval()
        self.extractor = extractor
        self.device = device
        self.logger = logger

    def predict(self, video_path):
        mat = self.extractor.process(video_path)  # (C, T)
        x = torch.FloatTensor(mat).unsqueeze(0).to(self.device)  # (1, C, T)
        with torch.no_grad():
            conc, risk_logits = self.model(x)
        risk_class = torch.argmax(risk_logits, dim=1).item()
        return {
            "concentration": conc.item(),
            "risk_logits": risk_logits.cpu().numpy().tolist()
        }