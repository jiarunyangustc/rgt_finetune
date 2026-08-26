import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryEncoder(nn.Module):
    """
    将 encoder 瓶颈层特征 (conv4, 512-ch) 压缩为紧凑的记忆条目。
    可选地融合当前 RGT 预测图，使记忆携带语义信息。

    SAM2 对应模块: memory_encoder
    """

    def __init__(self, in_channels: int = 512, mem_channels: int = 256):
        super().__init__()
        self.compress = nn.Sequential(
            nn.Conv2d(in_channels, mem_channels, kernel_size=1),
            nn.GroupNorm(8, mem_channels),
            nn.GELU(),
            nn.Conv2d(mem_channels, mem_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, mem_channels),
            nn.GELU(),
        )
        self.rgt_proj = nn.Conv2d(1, in_channels, kernel_size=1)

    def forward(self, feat: torch.Tensor,
                rgt_pred: torch.Tensor = None) -> torch.Tensor:
        """
        feat     : B × in_channels × H × W  (encoder 瓶颈特征 conv4)
        rgt_pred : B × 1 × H' × W'          (当前剖面 RGT 预测, 可选)
        returns  : B × mem_channels × H × W
        """
        if rgt_pred is not None:
            rgt_down = F.interpolate(
                rgt_pred, size=feat.shape[-2:],
                mode='bilinear', align_corners=False
            )
            feat = feat + self.rgt_proj(rgt_down)
        return self.compress(feat)


class MemoryAttentionLayer(nn.Module):
    """
    单层 Transformer: Self-Attn → Cross-Attn(→记忆库) → FFN
    结构与 SAM2 的 MemoryAttentionLayer 保持一致。
    """

    def __init__(self, d_model: int, mem_dim: int,
                 num_heads: int, mlp_ratio: float = 2.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads,
            kdim=mem_dim, vdim=mem_dim,
            batch_first=True
        )
        hidden = int(d_model * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        """
        x   : B × N × d_model   (当前剖面展平的空间 token)
        mem : B × M × mem_dim   (所有历史剖面拼接的记忆 token)
        """
        x2, _ = self.self_attn(x, x, x)
        x = self.norm1(x + x2)

        x2, _ = self.cross_attn(x, mem, mem)
        x = self.norm2(x + x2)

        x = self.norm3(x + self.ffn(x))
        return x


class MemoryAttention(nn.Module):
    """
    将当前剖面的 encoder 瓶颈特征以 Cross-Attention 方式条件化到
    历史剖面的记忆库上。

    SAM2 对应模块: memory_attention
    用法:
        conditioned_conv4 = memory_attn(conv4, memory_bank)
        其中 memory_bank 是外部维护的 list[Tensor]，每项形状
        为 B × mem_channels × H × W，由 MemoryEncoder 生成。
    """

    def __init__(self, feat_channels: int = 512, mem_channels: int = 256,
                 num_heads: int = 8, num_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([
            MemoryAttentionLayer(feat_channels, mem_channels, num_heads)
            for _ in range(num_layers)
        ])
        self.out_norm = nn.LayerNorm(feat_channels)
        # 可学习门控：初始化为 atanh(0.3)≈0.31，使 tanh(gate)≈0.3（30% memory 混入）
        # 若 memory 有害，gate 可学习到 0；若有益，gate 可增大到 1
        self.gate = nn.Parameter(torch.tensor([0.3095]))

    def forward(self, curr_feat: torch.Tensor,
                memory_bank: list) -> torch.Tensor:
        """
        curr_feat   : B × C × H × W
        memory_bank : list of (B × mem_channels × H × W), 长度 ≤ K
        returns     : B × C × H × W  (记忆条件化后的特征)
        """
        if not memory_bank:
            return curr_feat

        B, C, H, W = curr_feat.shape
        x = curr_feat.flatten(2).permute(0, 2, 1)          # B × HW × C

        mem_tokens = torch.cat(
            [m.flatten(2).permute(0, 2, 1) for m in memory_bank], dim=1
        )                                                    # B × (K*HW) × mem_C

        for layer in self.layers:
            x = layer(x, mem_tokens)

        x = self.out_norm(x)
        refined = x.permute(0, 2, 1).reshape(B, C, H, W)
        # 残差门控：gate=0 时完全保留原始 conv4，随训练逐渐引入 memory 修正量
        return curr_feat + torch.tanh(self.gate) * (refined - curr_feat)
