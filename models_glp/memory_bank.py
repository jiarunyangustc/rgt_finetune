import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryEncoder(nn.Module):
    """
    Compress encoder bottleneck features (conv4, 512 channels) into a compact
    memory entry. The current RGT prediction can optionally be fused into it.

    The design follows the memory-encoder role in SAM2.
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
        feat     : B x in_channels x H x W (encoder bottleneck features)
        rgt_pred : B x 1 x H' x W' (optional current-section RGT prediction)
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
    One transformer layer: self-attention, memory cross-attention, and FFN.
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
        x   : B x N x d_model (flattened current-section tokens)
        mem : B x M x mem_dim (concatenated memory tokens)
        """
        x2, _ = self.self_attn(x, x, x)
        x = self.norm1(x + x2)

        x2, _ = self.cross_attn(x, mem, mem)
        x = self.norm2(x + x2)

        x = self.norm3(x + self.ffn(x))
        return x


class MemoryAttention(nn.Module):
    """
    Condition the current encoder bottleneck features on previous-section
    memory entries through cross-attention.

    The design follows the memory-attention role in SAM2. Example:
        conditioned_conv4 = memory_attn(conv4, memory_bank)
        Here memory_bank is an externally maintained list of tensors produced
        by MemoryEncoder, each with shape B x mem_channels x H x W.
    """

    def __init__(self, feat_channels: int = 512, mem_channels: int = 256,
                 num_heads: int = 8, num_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([
            MemoryAttentionLayer(feat_channels, mem_channels, num_heads)
            for _ in range(num_layers)
        ])
        self.out_norm = nn.LayerNorm(feat_channels)


        self.gate = nn.Parameter(torch.tensor([0.3095]))

    def forward(self, curr_feat: torch.Tensor,
                memory_bank: list) -> torch.Tensor:
        """
        curr_feat   : B × C × H × W
        memory_bank : list of B x mem_channels x H x W tensors, length <= K
        returns     : B x C x H x W memory-conditioned features
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

        return curr_feat + torch.tanh(self.gate) * (refined - curr_feat)
