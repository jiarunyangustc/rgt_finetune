#!/usr/bin/env python
"""Paper-protocol horizon MAE for a predicted RGT volume.

Protocol (as described in the paper):
  1. For each picked horizon, assign one global RGT value: the median of the
     predicted RGT over all samples of the pick, taken over the whole volume
     (the median is invariant under monotonic rescaling of the prediction).
  2. On every section, make the predicted RGT non-decreasing along depth by a
     running maximum, read the depth of the assigned value per trace by linear
     interpolation of the inverse RGT-depth relation, and drop traces whose
     RGT range does not contain the value.
  3. Horizon MAE = per-section mean absolute depth difference to the pick,
     averaged over sections.  Group MAE (input / validation) = equal-weight
     average of the horizons in the group.

Usage:
  python eval_paper_protocol.py --volume pred.dat --shape 416,256,256 \
      --frame frame.dat --centers 0.2638,0.4061,0.7778,0.8445 \
      --input-idx 0,2 --valid-idx 1,3
Optional validation surfaces given as depth maps (float32, (n1, n2)):
  --valid-maps shallow.npy,mid.npy,deep.npy
Volumes are float32 with layout (n1, n2, depth); sections are indexed by n2.
"""
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--volume', required=True)
ap.add_argument('--shape', required=True, help='n1,n2,nz')
ap.add_argument('--frame', required=True, help='horizon frame volume (.dat)')
ap.add_argument('--centers', required=True, help='frame values of the horizons')
ap.add_argument('--input-idx', default='', help='indices (into centers) of input horizons')
ap.add_argument('--valid-idx', default='', help='indices of validation horizons')
ap.add_argument('--valid-maps', default='', help='extra validation depth maps (.npy, comma sep)')
ap.add_argument('--tol', type=float, default=1.5e-3)
a = ap.parse_args()

N1, N2, NZ = (int(v) for v in a.shape.split(','))
zg = np.arange(NZ, dtype=np.float64)
vol = np.fromfile(a.volume, np.float32).reshape(N1, N2, NZ)
PM = np.maximum.accumulate(vol.astype(np.float64), axis=2)
fr = np.fromfile(a.frame, np.float32).reshape(N1, N2, NZ)

maps, names = [], []
for k, c in enumerate(float(v) for v in a.centers.split(',')):
    m = np.abs(fr - c) < a.tol
    cnt = m.sum(axis=2)
    z = (m * zg[None, None, :]).sum(axis=2) / np.maximum(cnt, 1)
    z = np.where(cnt > 0, z, np.nan)
    z[(z < 2) | (z > NZ - 3)] = np.nan
    maps.append(z)
    names.append(f'h{k + 1}')
for p in filter(None, a.valid_maps.split(',')):
    maps.append(np.load(p).astype(np.float64))
    names.append(p)


def tau_at(Z):
    z = np.clip(np.nan_to_num(Z, nan=0.0), 0, NZ - 1 - 1e-6)
    z0 = np.floor(z).astype(int)
    f = z - z0
    i1, i2 = np.meshgrid(np.arange(N1), np.arange(N2), indexing='ij')
    pv = PM[i1, i2, z0] * (1 - f) + PM[i1, i2, np.minimum(z0 + 1, NZ - 1)] * f
    pv[~np.isfinite(Z)] = np.nan
    return pv


def iso_sec(pm, lv):
    nz, nt = pm.shape
    below = (pm < lv).sum(axis=0)
    z1 = np.clip(below, 1, nz - 1)
    p0, p1 = pm[z1 - 1, np.arange(nt)], pm[z1, np.arange(nt)]
    f = np.where(p1 > p0, (lv - p0) / np.maximum(p1 - p0, 1e-12), 0.0)
    zp = (z1 - 1 + np.clip(f, 0, 1)).astype(np.float64)
    zp[(lv < pm[0]) | (lv > pm[-1])] = np.nan
    return zp


maes = []
for nm, Z in zip(names, maps):
    lv = np.nanmedian(tau_at(Z))
    errs = []
    for j in range(N2):
        zp = iso_sec(PM[:, j, :].T, lv)
        d = np.abs(zp - Z[:, j])
        d = d[np.isfinite(d)]
        if d.size > N1 * 0.2:
            errs.append(d.mean())
    maes.append(np.mean(errs))
    print(f'{nm:24s} level {lv:.5f}  MAE {maes[-1]:.3f}')

def group(tag, idx_str, offset=0):
    if not idx_str:
        return
    idx = [int(v) + offset for v in idx_str.split(',')]
    print(f'{tag}: {np.mean([maes[i] for i in idx]):.3f} '
          f'(equal-weight over {[names[i] for i in idx]})')

group('INPUT  group MAE', a.input_idx)
group('VALID  group MAE', a.valid_idx)
