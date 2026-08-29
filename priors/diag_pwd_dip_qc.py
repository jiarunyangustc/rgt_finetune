#!/usr/bin/env python
"""Estimate native-grid cross-section PWD dips and perform quality control.

The shaping-regularized ``pyseistr.dip3dc`` implementation estimates the
stacking-axis slope in depth samples per section spacing. Its sign is aligned
with the reference slope ``dz/dy = -(du/dy)/(du/dz)`` and diagnostic error
statistics are reported before the dip field is saved.
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

    try:
        sx512 = np.load(SX512_PATH, mmap_mode='r')
        a = np.asarray(sx512[::2, ::2, ::2], dtype=np.float64).ravel()
        b = sx.astype(np.float64).ravel()
        r = np.corrcoef(a, b)[0, 1]
        print(f"[layout] correlation of sx_512[::2]^3 and sx_256 = {r:.4f}", flush=True)
    except Exception as e:
        print(f"[layout] skipped 512-volume check: {e}", flush=True)

    print("[pwd] running dip3dc on the native 256 cube (niter=5)", flush=True)
    din = np.ascontiguousarray(sx.transpose(2, 0, 1), dtype=np.float32)  # (z,x,y)
    _, dip_x = dip3dc(din, verb=0)
    dip = dip_x.transpose(1, 2, 0)

    u = np.fromfile(UX_PATH, dtype=np.single).reshape(256, 256, 256).astype(np.float64)
    gy = np.gradient(u, axis=1)
    gz = np.maximum(np.gradient(u, axis=2), 1e-4)
    st = -gy / gz


    r0 = np.corrcoef(dip.ravel()[::37], st.ravel()[::37])[0, 1]
    sgn = 1.0 if r0 >= 0 else -1.0
    dip = sgn * dip
    np.save(OUT_PATH, dip.astype(np.float32))
    print(f"[pwd] sign {'+' if sgn > 0 else '-'} (r0={r0:.3f}); saved {OUT_PATH}", flush=True)

    err = dip - st
    print(f"\n=== PWD versus reference stacking-axis slope ({st.size} samples) ===")
    r = np.corrcoef(dip.ravel()[::7], st.ravel()[::7])[0, 1]
    print(f"overall r = {r:.3f} | median |error| {np.median(np.abs(err)):.3f} "
          f"p90 {np.percentile(np.abs(err), 90):.3f}")

    print(f"\n{'reference bin':>12} {'n':>10} {'ref median':>11} {'PWD median':>12} "
          f"{'med|err|':>9} {'p90|err|':>9} {'sign match':>10}")
    ast = np.abs(st)
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (ast >= lo) & (ast < hi)
        if m.sum() < 1000:
            continue
        sgn_ok = (np.sign(dip[m]) == np.sign(st[m])) | (ast[m] < 0.1)
        print(f"{lo:>5.1f}-{hi:<6.1f} {m.sum():>10} {np.median(ast[m]):>9.2f} "
              f"{np.median(np.abs(dip[m])):>20.2f} {np.median(np.abs(err[m])):>9.3f} "
              f"{np.percentile(np.abs(err[m]), 90):>9.3f} {sgn_ok.mean()*100:>9.1f}%")

    print(f"\n{'depth band':>12} {'med|err|':>9} {'p90|err|':>9} {'r':>7}")
    for b in range(8):
        sl = slice(b * 32, (b + 1) * 32)
        e, d_, s_ = err[:, :, sl], dip[:, :, sl], st[:, :, sl]
        rb = np.corrcoef(d_.ravel()[::11], s_.ravel()[::11])[0, 1]
        print(f"{b*64:>5}-{(b+1)*64:<6} {np.median(np.abs(e)):>9.3f} "
              f"{np.percentile(np.abs(e), 90):>9.3f} {rb:>7.3f}")


if __name__ == '__main__':
    main()
