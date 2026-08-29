#!/usr/bin/env python
"""Render the Survey-1 input-section comparison used in the manuscript.

Rows show direct prediction and the cumulative addition of horizon, segment,
and structure constraints. Usage: python render_data1_2d_input.py [--notext]
[--no-arrows]. Output: qc_overlays/data1_2d_input_v2{_notext}.png.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans', 'DejaVu Sans']
NOTEXT = '--notext' in sys.argv
NOARROW = '--no-arrows' in sys.argv
if NOTEXT:
    from matplotlib.axes import Axes
    _noop = lambda self, *a, **k: None
    for _m in ('set_title', 'set_xlabel', 'set_ylabel', 'text'):
        setattr(Axes, _m, _noop)
import matplotlib.pyplot as plt
from matplotlib import cm, colors as mcolors
from matplotlib.patches import FancyArrow

L = '../data/lunnan/'
PKG = '../data/lunnan/'
CK = 'checkpoints/'
NIL, NXL, NZ = 416, 256, 256
IL = 163
WIN = (55, 115)
HZ_COLOR = ['#1e88e5', '#19a35b', '#ffab00', '#8e24aa']   # h1..h4
FITC = '#ff00ff'
SEG_COL = ['#4c72b0', '#dd8452', '#55a868', '#c44e52', '#8172b3', '#937860',
           '#da8bc3', '#64b5cd']
CYAN = '#00ffe1'
zg = np.arange(NZ, dtype=np.float64)


def last(d):
    import os
    fs = [f for f in os.listdir(CK + d) if f.endswith('.dat')]
    return CK + d + '/' + max(fs, key=lambda f: int(f.split('_')[1].split('.')[0]))


MODELS = [('Direct\nprediction', CK + 'lunnan_zeroshot_ln_fr_h13_filled.dat', False, False),
          ('+ Horizon\nconstraint', last('lunnan_h13_olhr_ep150_bs40'), False, False),
          ('+ Segment\nconstraint', last('lunnan_h13_segamp_only_ep150_bs40'), True, False),
          ('+ Structure\nconstraint', CK + 'lunnan_h13_segamp_pairdip_sr3_ep150_bs40/png_149.dat',
           True, True)]

print('[load]', flush=True)
sx = np.fromfile(L + 'ln_416x256x256.dat', np.float32).reshape(NIL, NXL, NZ)
sec = sx[IL].T; sec = (sec - sec.mean()) / sec.std()
fr = np.fromfile(L + 'ln_fr_wd3_x_rm.dat', np.float32).reshape(NIL, NXL, NZ)
CENT = [0.2638, 0.4061, 0.7778, 0.8445]
HZ = []
for c in CENT:
    m = np.abs(fr - c) < 1.5e-3
    cnt = m.sum(axis=2)
    HZ.append(np.where(cnt > 0, (m * zg[None, None, :]).sum(axis=2)
                       / np.maximum(cnt, 1), np.nan))
SEG = np.load(PKG + 'segment_pt_lunnan_tile_amp70.npy')
DIPI = np.load(PKG + 'pwd_dipi_lunnan.npy')               # (il,xl,z) dz/dxl


def isec(pm, lv):
    nz, nt = pm.shape
    below = (pm < lv).sum(axis=0)
    z1 = np.clip(below, 1, nz - 1)
    p0 = pm[z1 - 1, np.arange(nt)]
    p1 = pm[z1, np.arange(nt)]
    f = np.where(p1 > p0, (lv - p0) / np.maximum(p1 - p0, 1e-12), 0.0)
    zp = (z1 - 1 + np.clip(f, 0, 1)).astype(np.float64)
    zp[(lv < pm[0]) | (lv > pm[-1])] = np.nan
    return zp


def seg_polylines(seg_zx, minlen=10):
    lines = []
    for sid in np.unique(seg_zx):
        if sid <= 0:
            continue
        zs, xs = np.nonzero(seg_zx == sid)
        cnt = np.bincount(xs, minlength=NXL)
        if (cnt > 0).sum() < minlen:
            continue
        line = np.where(cnt > 0, np.bincount(xs, weights=zs, minlength=NXL)
                        / np.maximum(cnt, 1), np.nan)
        jump = np.nonzero(np.abs(np.diff(line)) > 4.0)[0]
        line[jump + 1] = np.nan
        lines.append(line)
    return lines


seg_lines = seg_polylines(SEG[IL].T.astype(np.int32))


def depth_cal(v_zx):
    m = np.maximum.accumulate(np.nanmean(np.maximum.accumulate(v_zx, axis=0), axis=1))
    m = m + np.arange(len(m)) * 1e-9
    return np.interp(v_zx, m, np.linspace(0, 1, len(m)))


try:
    import cigvis
    RGT_CMAP = cigvis.colormap.get_cmap_from_str('stratum')
except Exception:
    RGT_CMAP = 'jet'
cmap = cm.get_cmap('jet')

# per-model: sections, global levels, MAE
ROWS = []
for label, path, show_seg, show_dip in MODELS:
    vol = np.fromfile(path, np.float32).reshape(NIL, NXL, NZ)
    PM = [np.maximum.accumulate(vol[i].T.astype(np.float64), axis=0)
          for i in range(NIL)]
    levs = []
    for k in range(4):
        Z = HZ[k]; pv = []
        for i in range(NIL):
            zl = Z[i]
            pv.append(np.array([np.interp(zl[x], zg, PM[i][:, x])
                                if np.isfinite(zl[x]) and 0 <= zl[x] < NZ else np.nan
                                for x in range(NXL)]))
        levs.append(np.nanmedian(np.concatenate(pv)))
    pm = PM[IL]
    fits = [isec(pm, lv) for lv in levs]
    def mae(k):
        d = np.abs(fits[k] - HZ[k][IL]); d = d[np.isfinite(d)]
        return d.mean()
    inp, val = (mae(0) + mae(2)) / 2, (mae(1) + mae(3)) / 2
    w = np.abs(fits[3] - HZ[3][IL])[WIN[0]:WIN[1]]
    wmae = w[np.isfinite(w)].mean()
    seed = np.maximum.accumulate(vol[IL, 128, :].astype(np.float64))
    ROWS.append(dict(label=label, vol=vol, pm=pm, fits=fits, seed=seed,
                     show_seg=show_seg, show_dip=show_dip,
                     inp=inp, val=val, wmae=wmae))
    print(f'{label.split()[0]:10s} input {inp:.2f} valid {val:.2f} win {wmae:.1f}',
          flush=True)

best_in = min(r['inp'] for r in ROWS)
best_va = min(r['val'] for r in ROWS)

SEED_DEPTHS = [6, 25, 42, 60, 80, 100, 120, 140, 160, 180, 220, 240]
xg = np.arange(NXL)
fig, axes = plt.subplots(4, 5, figsize=(27.2, 4.2 * 4))
for r, R in enumerate(ROWS):
    pm = R['pm']
    # col1: inputs and constraints
    ax = axes[r, 0]
    ax.imshow(sec, cmap='gray', vmin=-2, vmax=2, aspect='auto',
              interpolation='bilinear')
    if R['show_seg']:
        for si, zm in enumerate(seg_lines):
            ax.plot(xg, zm, color='#111111', lw=4.2, alpha=0.75)
            ax.plot(xg, zm, color=SEG_COL[(si * 3) % 8], lw=2.8)
    if R['show_dip']:
        AL = 6.5
        for xq in range(10, NXL - 8, 18):
            for zq in range(10, NZ - 8, 18):
                d = float(DIPI[IL, xq, zq]); n = np.hypot(1.0, d)
                ax.plot([xq - AL / n, xq + AL / n],
                        [zq - AL * d / n, zq + AL * d / n],
                        color='k', lw=3.2, alpha=0.85, solid_capstyle='round')
                ax.plot([xq - AL / n, xq + AL / n],
                        [zq - AL * d / n, zq + AL * d / n],
                        color=CYAN, lw=1.8, solid_capstyle='round')
    for k in (0, 2):                                   # input horizons h1/h3
        ax.plot(xg, HZ[k][IL], color='k', lw=6.4, alpha=0.9)
        ax.plot(xg, HZ[k][IL], color=HZ_COLOR[k] if k else '#1e88e5', lw=4.2)
    ax.set_ylabel(R['label'], fontsize=17)
    if r == 0:
        ax.set_title('Inputs and\nconstraints', fontsize=17)
    # col2: predicted RGT
    ax = axes[r, 1]
    ax.imshow(depth_cal(sec * 0 + R['vol'][IL].T.astype(np.float64)),
              cmap=RGT_CMAP, vmin=0, vmax=1, aspect='auto',
              interpolation='bilinear')
    if r == 0:
        ax.set_title('Predicted\nRGT', fontsize=17)
    # col3: isochron overlay
    ax = axes[r, 2]
    ax.imshow(sec, cmap='gray', vmin=-2, vmax=2, aspect='auto',
              interpolation='bilinear')
    dnorm = mcolors.Normalize(vmin=min(SEED_DEPTHS), vmax=max(SEED_DEPTHS))
    for sd in SEED_DEPTHS:
        lv = np.interp(sd, zg, R['seed'])
        zp = isec(pm, lv)
        ax.plot(xg, zp, color=cmap(dnorm(sd)), lw=3.4, alpha=0.9)
    if not NOARROW:
        for (xa, za) in ((143, 52), (168, 62)):
            ax.add_patch(FancyArrow(xa + 9, za + 16, -7, -12, width=2.6,
                                    head_width=7.5, head_length=6.5,
                                    fc='#cc1111', ec='#3a0000', lw=1.0))
    if r == 0:
        ax.set_title('Isochron\noverlay', fontsize=17)
    # col4: input-horizon fit
    ax = axes[r, 3]
    ax.imshow(sec, cmap='gray', vmin=-2, vmax=2, aspect='auto',
              interpolation='bilinear')
    for k, fc in ((0, FITC), (2, 'black')):
        ax.plot(xg, HZ[k][IL], color=HZ_COLOR[0] if k == 0 else HZ_COLOR[2],
                lw=5.6, alpha=0.95)
        ax.plot(xg, R['fits'][k], ls=':', color=fc, lw=3.4)
    if not NOARROW:
        for (xa, za) in ((146, 72), (170, 80)):
            ax.add_patch(FancyArrow(xa - 8, za + 14, 6.5, -11, width=2.6,
                                    head_width=7.5, head_length=6.5,
                                    fc='#ffe600', ec='#4d4000', lw=1.0))
    if r == 0:
        ax.set_title('Input-\nhorizon fit', fontsize=17)
    # col5: validation fit + inset
    ax = axes[r, 4]
    ax.imshow(sec, cmap='gray', vmin=-2, vmax=2, aspect='auto',
              interpolation='bilinear')
    for k in (1, 3):
        ax.plot(xg, HZ[k][IL], color=HZ_COLOR[k], lw=5.6, alpha=0.95)
        ax.plot(xg, R['fits'][k], 'r:', lw=3.4)
    X0, X1 = WIN
    zl4 = HZ[3][IL]
    zc = np.nanmean(zl4[X0:X1]); ZH = 20
    ax.add_patch(plt.Rectangle((X0, zc - ZH), X1 - X0, 2 * ZH, fill=False,
                               ec='k', lw=2.0))
    axi = ax.inset_axes([0.05, 0.16, 0.42, 0.38])
    axi.imshow(sec, cmap='gray', vmin=-2, vmax=2, aspect='auto',
               interpolation='bilinear')
    axi.plot(xg, zl4, color=HZ_COLOR[3], lw=6.5)
    axi.plot(xg, R['fits'][3], 'r:', lw=4.2)
    axi.set_xlim(X0, X1); axi.set_ylim(zc + ZH, zc - ZH)
    axi.set_xticks([]); axi.set_yticks([])
    for sp in axi.spines.values():
        sp.set_linewidth(2.0)
    axi.text(0.06, 0.08, f'{R["wmae"]:.1f}', transform=axi.transAxes,
             fontsize=15, bbox=dict(fc='white', alpha=0.9, ec='none', pad=2))
    if r == 0:
        ax.set_title('Validation-\nhorizon fit', fontsize=17)
    ic = 'red' if abs(R['inp'] - best_in) < 5e-3 else 'black'
    vc = 'red' if abs(R['val'] - best_va) < 5e-3 else 'black'
    ax.text(1.05, 0.60, f'Input : {R["inp"]:.2f}', transform=ax.transAxes,
            fontsize=17, fontweight='bold', color=ic)
    ax.text(1.05, 0.42, f'Val. : {R["val"]:.2f}', transform=ax.transAxes,
            fontsize=17, fontweight='bold', color=vc)
    for c in range(5):
        axes[r, c].set_xlim(0, NXL - 1); axes[r, c].set_ylim(NZ - 1, 0)
        axes[r, c].set_xticks([]); axes[r, c].set_yticks([])

fig.tight_layout(rect=(0, 0, 0.92, 1))
out = 'qc_overlays/data1_2d_input_v2%s.png' % ('_notext' if NOTEXT else '')
fig.savefig(out, dpi=140, facecolor='white', bbox_inches='tight')
print('saved', out)
