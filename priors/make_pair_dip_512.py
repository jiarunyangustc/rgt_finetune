#!/usr/bin/env python
"""Upsample native-grid PWD dips and validate the warp convention.

The output has layout ``(section, depth, trace)``. ``dip[i]`` is the downward
shift from section ``i`` to ``i+1``, clipped to +/-4 depth samples. The unit
test verifies that dip compensation reduces the adjacent-section RGT
difference where the shift magnitude exceeds 0.5 samples and that reversing
the sign makes the alignment worse.
"""
import numpy as np
from scipy.ndimage import zoom

DIP_PATH = '../data/zxdata/pwd_dipx_256.npy'
UX_PATH = '../data/zxdata/ux_gh_256x256x256.dat'
OUT = '../data/zxdata/pair_dip_512.npy'


def warp_z(B, d):
    """Linearly sample B(z+d, x), using boundary values outside the range."""
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
        m = np.abs(dl) > 0.5
        if m.sum() < 100:
            continue
        d_same.append(np.abs(B - A)[m])
        d_warp.append(np.abs(warp_z(B, dl) - A)[m])
        d_flip.append(np.abs(warp_z(B, -dl) - A)[m])
    ds = np.concatenate(d_same); dw = np.concatenate(d_warp); df = np.concatenate(d_flip)
    print(f"[test] |delta|>0.5 subset n={ds.size}: same-position median {np.median(ds):.5f} | "
          f"warped {np.median(dw):.5f} | reversed {np.median(df):.5f}")
    ok = np.median(dw) < np.median(ds) < np.median(df)
    print(f"[test] {'passed: warped < same-position < reversed' if ok else 'failed: check axis and sign'}")
    return ok


def main():
    dip = np.load(DIP_PATH).astype(np.float32)
    if not unit_test(dip):
        raise SystemExit(1)
    dip512 = zoom(np.clip(dip, -4, 4), 2.0, order=1)[:512, :512, :512]
    out = np.ascontiguousarray(dip512.transpose(1, 2, 0))   # (y, z, x)
    np.save(OUT, out)
    print(f"[write] saved {OUT}, shape={out.shape} "
          f"|δ| med {np.median(np.abs(out)):.3f} p99 {np.percentile(np.abs(out),99):.3f}")


if __name__ == '__main__':
    main()
