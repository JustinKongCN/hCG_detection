import torch
import torch.nn as nn
import math


# ========== TCN 组件 ==========
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.utils.parametrizations.weight_norm(
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)),
            Chomp1d(padding), nn.ReLU(), nn.Dropout(dropout),
            nn.utils.parametrizations.weight_norm(
                nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)),
            Chomp1d(padding), nn.ReLU(), nn.Dropout(dropout)
        )
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


# ========== 位置编码 ==========
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0).transpose(0, 1))

    def forward(self, x):
        x = x.permute(2, 0, 1)
        x = x + self.pe[:x.size(0), :]
        return x.permute(1, 2, 0)


# ========== TCN + Transformer 融合模型 ==========
class hCG_TCN_MultiTask(nn.Module):
    def __init__(self, input_channels=8, tcn_filters=[64, 128],
                 d_model=128, nhead=4, num_layers=2,
                 num_classes=3, dropout=0.3, **kwargs):
        super().__init__()

        # 1. TCN
        layers = []
        for i, out_ch in enumerate(tcn_filters):
            in_ch = input_channels if i == 0 else tcn_filters[i - 1]
            layers.append(TemporalBlock(in_ch, out_ch, 3, 2 ** i, dropout))
        self.tcn = nn.Sequential(*layers)

        # 2. 投影到 Transformer 维度
        self.proj = nn.Conv1d(tcn_filters[-1], d_model, 1) if tcn_filters[-1] != d_model else nn.Identity()

        # 3. Transformer
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=False)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4. 输出头
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc_reg = nn.Linear(d_model, 1)
        self.fc_cls = nn.Linear(d_model, num_classes)

    def forward(self, x):
        out = self.tcn(x)
        out = self.proj(out)
        out = self.pos_encoder(out)
        out = out.permute(2, 0, 1)
        out = self.transformer(out)
        out = out.permute(1, 2, 0)
        out = self.pool(out).squeeze(-1)
        return self.fc_reg(out), self.fc_cls(out)