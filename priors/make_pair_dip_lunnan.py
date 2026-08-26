#!/usr/bin/env python
"""
lunnan δ 场: PWD 沿 inline(跨片)方向倾角 -> pair dip 体 + 纯地震质检
====================================================================
lunnan 无真值 RGT, 与 zxdata 版(make_pair_dip_512.py)差异:
  - 跨片方向 = il 轴(inline 方向预测, 切片=il 索引)
  - 符号校准: 与相邻片窗口互相关(NCC>0.7)的有符号移位对齐
  - warp 单测: 直接用地震——warp 后邻片振幅差应显著下降, 符号翻转应恶化
输出: ../data/lunnan/pair_dip_lunnan.npy, (il=416, z=256, x=xl=256) float32,
      dip[i] = 切片 i 与 i+1 间下移量 δ(z,x), 截断 ±4。
      约定同 zxdata: pred_{i+1}(z+δ,x) ≈ pred_i(z,x)。
附: lunnan 跨片斜率谱(补进论文工区谱表)。
"""
import numpy as np
from pyseistr import dip3dc

SX_PATH = '../data/lunnan/ln_416x256x256.dat'
OUT = '../data/lunnan/pair_dip_lunnan.npy'
NI, NX, NZ = 416, 256, 256
HW, LAG = 16, 8
rng = np.random.default_rng(0)


def xcorr_shift(sx, i, x, z):
    """il 片 i 与 i+1 在 (x,z) 的竖直移位(px, 正=下移), ncc; sx=(il,xl,z)"""
    if z - HW - LAG < 0 or z + HW + LAG >= NZ:
        return np.nan, 0.0
    a = sx[i, max(x-1, 0):x+2, z-HW:z+HW+1].mean(axis=0)
    a = a - a.mean(); na = np.linalg.norm(a) + 1e-9
    best, cc = 0, -2.0
    ccs = np.full(2*LAG+1, -2.0)
    for k, lag in enumerate(range(-LAG, LAG+1)):
        b = sx[i+1, max(x-1, 0):x+2, z-HW+lag:z+HW+1+lag].mean(axis=0)
        b = b - b.mean()
        c = float(a @ b / (na*(np.linalg.norm(b)+1e-9)))
        ccs[k] = c
        if c > cc:
            cc, best = c, lag
    k = best + LAG
    if 0 < k < 2*LAG:
        d2 = ccs[k-1] - 2*ccs[k] + ccs[k+1]
        if d2 < -1e-9:
            best = best + 0.5*(ccs[k-1]-ccs[k+1])/d2
    return float(best), cc


def main():
    sx = np.fromfile(SX_PATH, np.float32).reshape(NI, NX, NZ)
    print("[pwd] dip3dc (z,xl,il) 运行中...", flush=True)
    din = np.ascontiguousarray(sx.transpose(2, 1, 0), dtype=np.float32)   # (z, xl, il)
    _, dip_x = dip3dc(din, verb=0)            # 沿 n3=il 的斜率, (z, xl, il)
    dip = dip_x.transpose(2, 0, 1).astype(np.float64)                     # (il, z, xl)

    # 符号校准 + 幅度质检: 抽样点做互相关仲裁
    pts = np.stack([rng.integers(0, NI-1, 4000), rng.integers(1, NX-1, 4000),
                    rng.integers(HW+LAG+1, NZ-HW-LAG-1, 4000)], axis=1)
    rows = []
    for i, x, z in pts:
        s, cc = xcorr_shift(sx, i, x, z)
        if np.isfinite(s) and cc > 0.7:
            rows.append((dip[i, z, x], s))
    p, s = np.array(rows).T
    r0 = np.corrcoef(p, s)[0, 1]
    sgn = 1.0 if r0 >= 0 else -1.0
    dip *= sgn
    p *= sgn
    print(f"[xcorr] 有效 {len(rows)}/4000, corr(PWD,xcorr)={abs(r0):.3f} 符号取向 "
          f"{'+' if sgn > 0 else '-'} | xcorr |med| {np.median(np.abs(s)):.2f} "
          f"PWD |med| {np.median(np.abs(p)):.2f} |diff| med {np.median(np.abs(p-s)):.2f}")

    # 斜率谱(论文工区表用)
    a = np.abs(dip)
    print(f"[谱] lunnan (PWD, il向): med {np.median(a):.2f} p90 {np.percentile(a,90):.2f} "
          f"p99 {np.percentile(a,99):.2f} >1px {(a>1).mean()*100:.1f}% >2px {(a>2).mean()*100:.1f}%")

    # 地震 warp 单测: |δ|>0.5 处 warp 应降振幅差, 翻转应升
    z = np.arange(NZ)
    d_same, d_warp, d_flip = [], [], []
    for i in range(0, NI-1, 8):
        A, B = sx[i].T.astype(np.float64), sx[i+1].T.astype(np.float64)   # (z, xl)
        dl = dip[i]
        m = np.abs(dl) > 0.5
        if m.sum() < 100:
            continue
        for dd, acc in ((dl, d_warp), (-dl, d_flip)):
            zq = np.clip(z[:, None] + dd, 0, NZ-1-1e-6)
            z0 = zq.astype(int); f = zq - z0
            xi = np.broadcast_to(np.arange(NX)[None, :], B.shape)
            Bw = B[z0, xi]*(1-f) + B[np.minimum(z0+1, NZ-1), xi]*f
            acc.append(np.abs(Bw - A)[m])
        d_same.append(np.abs(B - A)[m])
    ds = np.concatenate(d_same); dw = np.concatenate(d_warp); df = np.concatenate(d_flip)
    print(f"[unittest] |δ|>0.5 n={ds.size}: 同位置 {np.median(ds):.4f} | warp {np.median(dw):.4f} "
          f"| 翻转 {np.median(df):.4f} -> {'通过' if np.median(dw) < np.median(ds) < np.median(df) else '失败!'}")
    if not (np.median(dw) < np.median(ds) < np.median(df)):
        raise SystemExit(1)

    np.save(OUT, np.clip(dip, -4, 4).astype(np.float32))
    print(f"[make] 已存 {OUT} shape={dip.shape} (il, z, xl)")


if __name__ == '__main__':
    main()
