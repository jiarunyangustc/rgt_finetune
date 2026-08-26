#!/usr/bin/env python
"""
δ 场预计算: PWD 跨片倾角 -> 512 训练用 pair dip 体 + warp 方向单元测试
=====================================================================
输入: pwd_dipx_256.npy (x,y,z 原生, px/片, 已 QC: 与地震互相关仲裁一致)
输出: pair_dip_512.npy, 布局 (y=xline切片, z, x) 512^3 float32,
      dip[i] = 切片 i 与 i+1 之间的下移量 δ(z,x), 值域截断 ±4。
      数值约定: px512 / 512片距 (与原生 px/片 数值相等, 已推导)。
      warp 约定: pred_{i+1}(z + δ, x) ≈ pred_i(z, x)。

单元测试(关键, δ 仅 ~0.4px, 符号/朝向错了等价性实验会"假通过"):
用 ux_gh 真值验证 warp 后邻片差在 |δ| 大的子集上应显著小于同位置差,
且符号翻转版应变差。
"""
import numpy as np
from scipy.ndimage import zoom

DIP_PATH = '../data/zxdata/pwd_dipx_256.npy'
UX_PATH = '../data/zxdata/ux_gh_256x256x256.dat'
OUT = '../data/zxdata/pair_dip_512.npy'


def warp_z(B, d):
    """B,d: (z,x); 返回 B(z+d, x) 线性插值, 越界=边界值"""
    Z = B.shape[0]
    z = np.arange(Z)[:, None] + d
    z0 = np.clip(np.floor(z).astype(int), 0, Z - 1)
    z1 = np.clip(z0 + 1, 0, Z - 1)
    f = z - z0
    xi = np.broadcast_to(np.arange(B.shape[1])[None, :], B.shape)
    return B[z0, xi] * (1 - f) + B[z1, xi] * f


def unit_test(dip):
    u = np.fromfile(UX_PATH, dtype=np.single).reshape(256, 256, 256).astype(np.float64)
    d_same, d_warp, d_flip = [], [], []
    for y in range(0, 255, 8):
        A = u[:, y, :].T
        B = u[:, y + 1, :].T
        dl = dip[:, y, :].T
        m = np.abs(dl) > 0.5                     # 只在 δ 有实质意义处考核
        if m.sum() < 100:
            continue
        d_same.append(np.abs(B - A)[m])
        d_warp.append(np.abs(warp_z(B, dl) - A)[m])
        d_flip.append(np.abs(warp_z(B, -dl) - A)[m])
    ds = np.concatenate(d_same); dw = np.concatenate(d_warp); df = np.concatenate(d_flip)
    print(f"[unittest] |δ|>0.5 子集 n={ds.size}: 同位置差 med {np.median(ds):.5f} | "
          f"warp后 {np.median(dw):.5f} | 符号翻转 {np.median(df):.5f}")
    ok = np.median(dw) < np.median(ds) < np.median(df)
    print(f"[unittest] {'通过: warp<同位置<翻转' if ok else '!!! 失败: 朝向或符号有问题'}")
    return ok


def main():
    dip = np.load(DIP_PATH).astype(np.float32)              # (x, y, z) 原生
    if not unit_test(dip):
        raise SystemExit(1)
    dip512 = zoom(np.clip(dip, -4, 4), 2.0, order=1)[:512, :512, :512]
    out = np.ascontiguousarray(dip512.transpose(1, 2, 0))   # (y, z, x)
    np.save(OUT, out)
    print(f"[make] 已存 {OUT} shape={out.shape} "
          f"|δ| med {np.median(np.abs(out)):.3f} p99 {np.percentile(np.abs(out),99):.3f}")


if __name__ == '__main__':
    main()
