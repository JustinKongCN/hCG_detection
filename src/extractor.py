import cv2
import numpy as np
from scipy.signal import find_peaks

class HCG_FeatureExtractor:
    def __init__(self, roi_box, fps_target, **kwargs):
        self.roi = roi_box
        self.fps = fps_target
        self.C_y_ratio = 0.30
        self.T_y_ratio = 0.65

    def _extract_frame(self, frame):
        x1, y1, x2, y2 = self.roi
        roi = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # 1. 全局RGB均值
        R = np.mean(roi[:, :, 2])
        G = np.mean(roi[:, :, 1])
        B = np.mean(roi[:, :, 0])

        # 2. 投影峰值法提取 C/T 线强度（核心改进）
        C_intensity, T_intensity = self._extract_bands_by_projection(gray)

        # 3. T/C 比值（消除批次/光照差异）
        TC_ratio = T_intensity / (C_intensity + 1e-6)

        # 4. 背景噪声（两侧边缘均值）
        bg = (np.mean(gray[:, :w // 10]) + np.mean(gray[:, -w // 10:])) / 2

        # 5. 层析前沿位置（纵向亮度梯度最大处）
        col_profile = np.mean(gray, axis=1)
        front = 0
        if len(col_profile) > 1:
            gradient = np.abs(np.diff(col_profile))
            front = int(np.argmax(gradient))

        return np.array([R, G, B, T_intensity, C_intensity, TC_ratio, bg, front])

    def _extract_bands_by_projection(self, gray):
        """投影峰值法：纵向灰度投影 + 自适应找峰"""
        h, w = gray.shape
        
        # 纵向投影：每行像素的平均灰度
        row_profile = np.mean(gray, axis=1)  # 形状 (h,)
        
        # 先验位置（用于峰分配和fallback）
        c_prior = int(h * self.C_y_ratio)
        t_prior = int(h * self.T_y_ratio)
        
        # 自适应 prominence：基于边缘背景估计噪声水平
        edge = np.concatenate([row_profile[:h//10], row_profile[-h//10:]])
        noise_std = np.std(edge)
        prominence = max(8.0, noise_std * 2.5)  # 至少8，或噪声的2.5倍
        
        # 找峰（峰间距至少为 h*0.15，防止把一条粗带拆成两个峰）
        min_distance = max(20, int(h * 0.15))
        peaks, properties = find_peaks(row_profile, distance=min_distance, prominence=prominence)
        
        # 分配峰到 C/T 线
        if len(peaks) >= 2:
            # 取最显著的两个峰
            prom = properties['prominences']
            top2_idx = np.argsort(prom)[-2:]
            top2_peaks = peaks[top2_idx]
            
            # 按纵向位置排序（y小的靠上）
            top2_sorted = np.sort(top2_peaks)
            upper, lower = top2_sorted[0], top2_sorted[1]
            
            # 按距离先验位置分配：upper离0.3近→C，lower离0.65近→T
            if abs(upper - c_prior) < abs(upper - t_prior):
                c_peak, t_peak = upper, lower
            else:
                c_peak, t_peak = lower, upper
                
            return float(row_profile[c_peak]), float(row_profile[t_peak])
        
        elif len(peaks) == 1:
            # 单峰fallback：检测到的峰指定为T线，C线回退到固定位置
            return float(row_profile[c_prior]), float(row_profile[peaks[0]])
        
        else:
            # 无峰fallback：回退到固定位置法
            return float(row_profile[c_prior]), float(row_profile[t_prior])

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
