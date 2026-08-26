#!/usr/bin/env python
"""network.png 重绘(修正版): LoRA + 选定卷积微调配置示意
修正: stage3 SRConv=Frozen; stage4 无 SRConv; decoder=Frozen(灰);
输入=地震剖面+层位通道。输出 qc_overlays/network_v2.png
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans', 'DejaVu Sans']
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib import cm

GREEN, GEDGE = '#c8e6c9', '#2e7d32'
BLUE, BEDGE = '#bbdefb', '#1565c0'
GRAY, GREDGE = '#e6e6e6', '#8a8a8a'
LAV = '#f6f6fc'

fig = plt.figure(figsize=(20, 11))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')


def box(x, y, w, h, fc, ec, txt='', fs=13, lw=1.6, style='round,pad=0.004',
        ls='solid', bold=False, tc='black'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec,
                                lw=lw, ls=ls, mutation_aspect=20 / 11))
    if txt:
        ax.text(x + w / 2, y + h / 2, txt, ha='center', va='center',
                fontsize=fs, fontweight='bold' if bold else 'normal', color=tc)


def arrow(p0, p1, lw=2.4, color='black'):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=22,
                                 lw=lw, color=color, shrinkA=0, shrinkB=0))


# ================= 顶部: 主流程 =================
TOP_Y0, TOP_H = 0.585, 0.375
# --- 编码器容器 ---
EX0, EX1 = 0.185, 0.660
box(EX0 - 0.012, TOP_Y0 - 0.015, EX1 - EX0 + 0.024, TOP_H + 0.03, LAV, '#3949ab',
    lw=1.8, ls=(0, (5, 4)))
ax.text((EX0 + EX1) / 2, TOP_Y0 + TOP_H - 0.005, 'Pretrained GLPN encoder (MiT)',
        ha='center', va='center', fontsize=19, fontweight='bold')

STAGES = [('Stage 1', '(1/4)'), ('Stage 2', '(1/8)'),
          ('Stage 3', '(1/16)'), ('Stage 4', '(1/32)')]
SW, GAP = 0.098, 0.022
BH, BGAP = 0.052, 0.0165
ROWS = ['att', 'sr', 'ffn', 'dw']
ROW_Y = {r: TOP_Y0 + TOP_H - 0.125 - i * (BH + BGAP) for i, r in enumerate(ROWS)}
stage_x = [EX0 + i * (SW + GAP) for i in range(4)]
for i, (name, frac) in enumerate(STAGES):
    x = stage_x[i]
    ax.text(x + SW / 2, TOP_Y0 + TOP_H - 0.052, f'{name} {frac}',
            ha='center', va='center', fontsize=14.5, fontweight='bold')
    box(x, ROW_Y['att'], SW, BH, GREEN, GEDGE, 'Attention\n(LoRA)', fs=12)
    if i <= 2:
        box(x, ROW_Y['sr'], SW, BH, BLUE, BEDGE, 'SRConv\n(Full FT)', fs=12)
    else:
        box(x, ROW_Y['sr'], SW, BH, 'white', GREDGE, 'no SRConv', fs=11.5,
            ls=(0, (3, 3)), tc='#666666')
    box(x, ROW_Y['ffn'], SW, BH, GREEN, GEDGE, 'Mix-FFN\n(LoRA)', fs=12)
    box(x, ROW_Y['dw'], SW, BH, BLUE, BEDGE, 'DWConv\n(Full FT)', fs=12)
mid_y = (ROW_Y['sr'] + ROW_Y['ffn'] + BH) / 2
for i in range(3):
    arrow((stage_x[i] + SW + 0.002, mid_y), (stage_x[i + 1] - 0.002, mid_y))

# --- 输入缩略图: 地震 + 层位通道 ---
D = '../data/zxdata/'
N = 256
sx = np.fromfile(D + 'sx_256x256x256.dat', np.float32).reshape(N, N, N)
sec = sx[:, 100, :].T                     # (z, x)
sec = (sec - sec.mean()) / sec.std()
FR = np.fromfile(D + 'frame_256_256_256_hr4_gh.dat', np.float32).reshape(N, N, N)
frs = FR[:, 100, :].T
axi = fig.add_axes([0.028, 0.655, 0.115, 0.24])
axi.imshow(sec, cmap='gray', vmin=-2, vmax=2, aspect='auto')
HZC = ['#2e86ff', '#19c37d', '#ffa62b', '#c355f5']
for c, colr in zip([0.0438, 0.2276, 0.6193, 0.8446], HZC):
    m = np.abs(frs - c) < 1.5e-3
    zs, xs = np.nonzero(m)
    if zs.size:
        cnt = np.bincount(xs, minlength=N).astype(float)
        zl = np.where(cnt > 0, np.bincount(xs, weights=zs, minlength=N)
                      / np.maximum(cnt, 1), np.nan)
        axi.plot(np.arange(N), zl, color=colr, lw=2.6)
axi.set_xticks([]); axi.set_yticks([])
for sp in axi.spines.values():
    sp.set_linewidth(1.4)
ax.text(0.0855, 0.925, 'Seismic section\n+ horizon channel', ha='center',
        va='center', fontsize=15, fontweight='bold')
arrow((0.146, mid_y), (EX0 - 0.014, mid_y))

# --- 解码器(冻结) 与输出 ---
DX = EX1 + 0.028
box(DX, mid_y - 0.085, 0.085, 0.17, GRAY, GREDGE, 'GLPN\ndecoder\n(Frozen)',
    fs=14.5, lw=1.8)
arrow((EX1 + 0.012, mid_y), (DX - 0.002, mid_y))
rgt = np.fromfile('checkpoints/'
                  'zxdata_hr4_3d_xline256_lossonly_pairdip_ep150_bs40/png_149.dat',
                  np.float32).reshape(N, N, N)
rsec = rgt[:, 100, :].T.astype(np.float64)
prof = np.maximum.accumulate(np.maximum.accumulate(rsec, axis=0).mean(axis=1))
prof = prof + np.arange(len(prof)) * 1e-9
cal = np.interp(rsec, prof, np.linspace(0, 1, N))
axo = fig.add_axes([0.795, 0.655, 0.115, 0.24])
axo.imshow(cal, cmap='jet', vmin=0, vmax=1, aspect='auto')
axo.set_xticks([]); axo.set_yticks([])
for sp in axo.spines.values():
    sp.set_linewidth(1.4)
ax.text(0.8525, 0.925, 'RGT prediction', ha='center', va='center',
        fontsize=15, fontweight='bold')
arrow((DX + 0.087, mid_y), (0.793, mid_y))

# --- 图例 ---
LX, LY = 0.775, 0.535
for k, (fc, ec, t) in enumerate([(GREEN, GEDGE, 'LoRA fine-tuning (trainable)'),
                                 (BLUE, BEDGE, 'Full fine-tuning (trainable)'),
                                 (GRAY, GREDGE, 'Frozen (not trainable)')]):
    y = LY - k * 0.045
    box(LX, y, 0.022, 0.028, fc, ec)
    ax.text(LX + 0.032, y + 0.014, t, ha='left', va='center', fontsize=13.5)
box(LX - 0.015, LY - 2 * 0.045 - 0.016, 0.235, 0.145, 'none', '#9e9e9e', lw=1.2)

# ================= 底部: zoom-in =================
ZX0, ZY0, ZW, ZH = 0.028, 0.045, 0.70, 0.44
box(ZX0, ZY0, ZW, ZH, 'none', '#1e88e5', lw=1.6, ls=(0, (6, 4)))
ax.text(ZX0 + 0.012, ZY0 + ZH - 0.035,
        '(b) Zoom in: one MiT transformer block '
        '(the SRConv is absent in stage 4, where sr\u2009ratio\u2009=\u20091)',
        ha='left', va='center', fontsize=15.5, color='#1565c0', fontweight='bold')
zline = ZY0 + ZH / 2 - 0.035
# Pre-Norm
box(ZX0 + 0.018, zline - 0.030, 0.055, 0.062, GRAY, GREDGE, 'Pre-Norm\n(Frozen)', fs=11.5)
arrow((ZX0 + 0.073, zline), (ZX0 + 0.090, zline), lw=1.8)
# attention container
AX0, AW = ZX0 + 0.090, 0.245
box(AX0, ZY0 + 0.05, AW, ZH - 0.155, '#eef7ee', GEDGE, lw=1.6)
ax.text(AX0 + AW / 2, ZY0 + ZH - 0.125, 'Multi-head attention', ha='center',
        va='center', fontsize=14, fontweight='bold')
qy = [zline + 0.075, zline + 0.005, zline - 0.065]
for t, y in zip(['Q Linear  (+LoRA)', 'K Linear  (+LoRA)', 'V Linear  (+LoRA)'], qy):
    box(AX0 + 0.015, y, 0.115, 0.052, GREEN, GEDGE, t, fs=11)
box(AX0 + 0.015, zline - 0.135, 0.115, 0.052, GREEN, GEDGE, 'Proj Linear  (+LoRA)', fs=11)
box(AX0 + 0.155, zline - 0.032, 0.072, 0.066, BLUE, BEDGE, 'SRConv\n(Full FT)', fs=11.5)
for y in qy[1:]:
    arrow((AX0 + 0.130, y + 0.026), (AX0 + 0.153, zline + 0.001), lw=1.4)
plus1 = (AX0 + AW + 0.028, zline)
ax.add_patch(plt.Circle(plus1, 0.011, fc='white', ec='black', lw=1.6))
ax.text(*plus1, '+', ha='center', va='center', fontsize=15)
arrow((AX0 + AW, zline), (plus1[0] - 0.012, zline), lw=1.8)
# second pre-norm
PN2 = plus1[0] + 0.025
box(PN2, zline - 0.030, 0.055, 0.062, GRAY, GREDGE, 'Pre-Norm\n(Frozen)', fs=11.5)
arrow((plus1[0] + 0.012, zline), (PN2 - 0.002, zline), lw=1.8)
# Mix-FFN container
FX0, FW = PN2 + 0.072, 0.155
box(FX0, ZY0 + 0.05, FW, ZH - 0.155, '#eaf3fd', BEDGE, lw=1.6)
ax.text(FX0 + FW / 2, ZY0 + ZH - 0.125, 'Mix-FFN', ha='center', va='center',
        fontsize=14, fontweight='bold')
ffy = [zline + 0.075, zline + 0.005, zline - 0.065, zline - 0.135]
for (t, fc, ec), y in zip([('FC1 Linear  (+LoRA)', GREEN, GEDGE),
                           ('DWConv  (Full FT)', BLUE, BEDGE),
                           ('GELU', GRAY, GREDGE),
                           ('FC2 Linear  (+LoRA)', GREEN, GEDGE)], ffy):
    box(FX0 + 0.015, y, FW - 0.030, 0.052, fc, ec, t, fs=11)
    if y != ffy[-1]:
        arrow((FX0 + FW / 2, y - 0.001), (FX0 + FW / 2, y - 0.016), lw=1.4)
arrow((PN2 + 0.055, zline), (FX0 - 0.002, zline), lw=1.8)
plus2 = (FX0 + FW + 0.028, zline)
ax.add_patch(plt.Circle(plus2, 0.011, fc='white', ec='black', lw=1.6))
ax.text(*plus2, '+', ha='center', va='center', fontsize=15)
arrow((FX0 + FW, zline), (plus2[0] - 0.012, zline), lw=1.8)
arrow((plus2[0] + 0.012, zline), (plus2[0] + 0.045, zline), lw=1.8)
# zoom guide lines from stage1/2 to panel
for xg in (stage_x[0] + SW / 2, stage_x[1] + SW / 2):
    ax.plot([xg, ZX0 + ZW * 0.35], [TOP_Y0 - 0.017, ZY0 + ZH],
            ls=(0, (4, 4)), lw=1.2, color='#1e88e5')

fig.savefig('qc_overlays/network_v2.png', dpi=160, facecolor='white')
print('saved qc_overlays/network_v2.png')
