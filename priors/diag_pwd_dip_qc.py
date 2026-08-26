#!/usr/bin/env python
"""
PWD 跨片倾角估计器"上岗证"（零 GPU）
=====================================
用 pyseistr 的 shaping 正则化 PWD (dip3dc) 在原生 256 地震 sx_256.npy 上估
跨片(xline)方向斜率 δ(px/片)，与 ux_gh 真值 RGT 的等值面跨片斜率逐点对比。
通过则 δ 场可进 pair loss 的 dip 补偿 warp。

真值斜率(有符号): dz/dy = -(∂u/∂y)/(∂u/∂z)，u 逐列单调增 → ∂u/∂z>0。
布局: sx/ux 均为 (x, xline, z)；dip3dc 输入 (z, x, y=xline)。

用法: python diag_pwd_dip_qc.py   (seismic_master 下, py39)
输出: ../data/zxdata/pwd_dipx_256.npy (布局同 ux, (x,y,z), 符号=真值约定)
"""
import numpy as np
from pyseistr import dip3dc

SX_PATH = '../data/zxdata/sx_256.npy'
SX512_PATH = '../data/zxdata/sx_512x512x512.npy'
UX_PATH = '../data/zxdata/ux_gh_256x256x256.dat'
OUT_PATH = '../data/zxdata/pwd_dipx_256.npy'
BINS = [0.0, 0.5, 1.0, 2.0, 4.0, np.inf]


def main():
    sx = np.load(SX_PATH)
    if sx.shape != (256, 256, 256):
        sx = sx.reshape(256, 256, 256)
    # 布局核验: 与 512 体降采样对齐 (同布局则相关应接近 1)
    try:
        sx512 = np.load(SX512_PATH, mmap_mode='r')
        a = np.asarray(sx512[::2, ::2, ::2], dtype=np.float64).ravel()
        b = sx.astype(np.float64).ravel()
        r = np.corrcoef(a, b)[0, 1]
        print(f"[layout] sx_512[::2]^3 vs sx_256 相关 = {r:.4f}", flush=True)
    except Exception as e:
        print(f"[layout] 512 体核验跳过: {e}", flush=True)

    print("[pwd] dip3dc 运行中 (256^3, niter=5)...", flush=True)
    din = np.ascontiguousarray(sx.transpose(2, 0, 1), dtype=np.float32)  # (z,x,y)
    _, dip_x = dip3dc(din, verb=0)          # (z,x,y), z采样点/xline道
    dip = dip_x.transpose(1, 2, 0)          # 回 (x,y,z)

    u = np.fromfile(UX_PATH, dtype=np.single).reshape(256, 256, 256).astype(np.float64)
    gy = np.gradient(u, axis=1)
    gz = np.maximum(np.gradient(u, axis=2), 1e-4)
    st = -gy / gz                            # 真值有符号斜率 px/片

    # PWD 符号约定与真值对齐(整体相关取向)
    r0 = np.corrcoef(dip.ravel()[::37], st.ravel()[::37])[0, 1]
    sgn = 1.0 if r0 >= 0 else -1.0
    dip = sgn * dip
    np.save(OUT_PATH, dip.astype(np.float32))
    print(f"[pwd] 符号取向 {'+' if sgn > 0 else '-'} (r0={r0:.3f}), 已存 {OUT_PATH}", flush=True)

    err = dip - st
    print(f"\n=== PWD vs ux_gh 真值跨片斜率 (px/片, 全体 {st.size} 体素) ===")
    r = np.corrcoef(dip.ravel()[::7], st.ravel()[::7])[0, 1]
    print(f"整体相关 r = {r:.3f} | 误差 med|e| {np.median(np.abs(err)):.3f} "
          f"p90 {np.percentile(np.abs(err), 90):.3f}")

    print(f"\n{'真值|斜率|箱':>12} {'n':>10} {'真值med':>9} {'PWD med(带符号追踪)':>20} "
          f"{'med|err|':>9} {'p90|err|':>9} {'符号一致率':>10}")
    ast = np.abs(st)
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (ast >= lo) & (ast < hi)
        if m.sum() < 1000:
            continue
        sgn_ok = (np.sign(dip[m]) == np.sign(st[m])) | (ast[m] < 0.1)
        print(f"{lo:>5.1f}-{hi:<6.1f} {m.sum():>10} {np.median(ast[m]):>9.2f} "
              f"{np.median(np.abs(dip[m])):>20.2f} {np.median(np.abs(err[m])):>9.3f} "
              f"{np.percentile(np.abs(err[m]), 90):>9.3f} {sgn_ok.mean()*100:>9.1f}%")

    print(f"\n{'深度带(512px)':>12} {'med|err|':>9} {'p90|err|':>9} {'r':>7}")
    for b in range(8):
        sl = slice(b * 32, (b + 1) * 32)
        e, d_, s_ = err[:, :, sl], dip[:, :, sl], st[:, :, sl]
        rb = np.corrcoef(d_.ravel()[::11], s_.ravel()[::11])[0, 1]
        print(f"{b*64:>5}-{(b+1)*64:<6} {np.median(np.abs(e)):>9.3f} "
              f"{np.percentile(np.abs(e), 90):>9.3f} {rb:>7.3f}")


if __name__ == '__main__':
    main()
