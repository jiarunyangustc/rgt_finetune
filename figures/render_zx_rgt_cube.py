#!/usr/bin/env python
"""zxdata 128^3 角块 RGT 渲染(workflow 最右输出格素材, 与 dip_cube 同窗同视角)
用法: python render_zx_rgt_cube.py [--origin 64,64,64] [--size 128]
              [--cmap jet] [--rgt <path>]
输出: qc_overlays/zx_rgt_cube_<size>_<cmap>.png (白底, 无文字)
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

ap = argparse.ArgumentParser()
ap.add_argument('--origin', default='64,64,64')
ap.add_argument('--size', type=int, default=128)
ap.add_argument('--cmap', default='jet')
ap.add_argument('--rgt', default='checkpoints/zxdata_hr4_3d_xline256_lossonly_pairdip_ep150_bs40/png_149.dat')
a = ap.parse_args()
X0, Y0, Z0 = (int(v) for v in a.origin.split(','))
M = a.size
X1, Y1, Z1 = X0 + M, Y0 + M, Z0 + M

rg = np.fromfile(a.rgt, np.float32).reshape(256, 256, 256).astype(np.float64)
crop = np.maximum.accumulate(rg[X0:X1, Y0:Y1, Z0:Z1], axis=2)
# 深度校准: 让色带沿深度均匀(单调重映射, 等值面不变)
prof = np.maximum.accumulate(crop.mean(axis=(0, 1)))
zt = np.linspace(0, 1, M)
cal = np.interp(rg, prof, zt)          # 整体重映射, 三面取值一致

xs, ys, zs = np.arange(X0, X1), np.arange(Y0, Y1), np.arange(Z0, Z1)


def faces(vol):
    Yg, Zg = np.meshgrid(ys, zs, indexing='ij')
    f1 = (np.full_like(Yg, X1 - 1, float), Yg * 1., Zg * 1., vol[X1 - 1, Y0:Y1, Z0:Z1])
    Xg, Zg2 = np.meshgrid(xs, zs, indexing='ij')
    f2 = (Xg * 1., np.full_like(Xg, Y0, float), Zg2 * 1., vol[X0:X1, Y0, Z0:Z1])
    Xg2, Yg2 = np.meshgrid(xs, ys, indexing='ij')
    f3 = (Xg2 * 1., Yg2 * 1., np.full_like(Xg2, Z0, float), vol[X0:X1, Y0:Y1, Z0])
    return [f1, f2, f3]


fig = plt.figure(figsize=(7.2, 6.8))
ax = fig.add_subplot(111, projection='3d')
mp = cm.ScalarMappable(plt.Normalize(0, 1), a.cmap)
for (Xf, Yf, Zf, V) in faces(cal):
    ax.plot_surface(Xf, Yf, Zf, facecolors=mp.to_rgba(V), rstride=1, cstride=1,
                    linewidth=0, antialiased=False, shade=False)
ax.set_xlim(X0, X1); ax.set_ylim(Y0, Y1); ax.set_zlim(Z1, Z0)
ax.view_init(elev=22, azim=-56)
ax.set_box_aspect((1, 1, 0.95))
ax.set_axis_off()
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
out = f'qc_overlays/zx_rgt_cube_{M}_{a.cmap}.png'
fig.savefig(out, dpi=170, facecolor='white')
print('saved', out)
