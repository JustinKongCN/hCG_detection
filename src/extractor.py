import cv2
import numpy as np
from pathlib import Path

class HCG_FeatureExtractor:
    def __init__(self, roi_box, fps_target, **kwargs):
        self.roi = roi_box
        self.fps = fps_target
        self.C_y_ratio = 0.30
        self.T_y_ratio = 0.65
        self.band_h = 20

    def _extract_frame(self, frame):
        x1, y1, x2, y2 = self.roi
        roi = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        R = np.mean(roi[:, :, 2])
        G = np.mean(roi[:, :, 1])
        B = np.mean(roi[:, :, 0])

        Cy = int(h * self.C_y_ratio)
        Ty = int(h * self.T_y_ratio)
        C_int = np.mean(gray[Cy - self.band_h:Cy + self.band_h, :])
        T_int = np.mean(gray[Ty - self.band_h:Ty + self.band_h, :])

        bg = (np.mean(gray[:, :w // 10]) + np.mean(gray[:, -w // 10:])) / 2
        front = 0  # 简化版，后续再完善

        return np.array([R, G, B, T_int, C_int, T_int / (C_int + 1e-6), bg, front])

    def process(self, video_path):
        cap = cv2.VideoCapture(str(video_path))
        fps_src = cap.get(cv2.CAP_PROP_FPS)
        interval = int(fps_src / self.fps)
        feats = []
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % interval == 0:
                feats.append(self._extract_frame(frame))
            idx += 1
        cap.release()
        mat = np.array(feats).T  # (C, T)
        mat = (mat - mat.mean(axis=1, keepdims=True)) / (mat.std(axis=1, keepdims=True) + 1e-6)
        return mat