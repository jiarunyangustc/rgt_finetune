#!/usr/bin/env python
"""zxdata size-pair figures, chain512_v2T-compatible 5-column layout.
Fig A (input section)      qc_overlays/zx_sizepair_inputsec.png
Fig B (orthogonal section) qc_overlays/zx_sizepair_orthosec.png
Rows: 256^3 no-FT / 256^3 full FT (pairdip) / 512^3 no-FT / 512^3 full FT (pairdip)
Cols: inputs (horizons; + segments & structural dips on full rows) |
      predicted RGT | prediction (RGT iso-lines) | fit to input horizons |
      validation-horizon fit (zoom inset on the deep horizon)
MAE labels = paper protocol (global level per horizon, whole volume; error on
the displayed section; equal-weight groups). Volume MAE printed to stdout.
Validation truths: hr1 pick (shallow) + ux_gh 0.3502 / 0.8443 (mid / deep).
Sections = the established display sections of chain512_v2T (512:
xline 3 / inline 157) and their half-index counterparts on 256
(xline 2 / inline 78). Iso-line anchors follow chain512_v2T.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
NOTEXT = '--notext' in sys.argv
if NOTEXT:
    from matplotlib.axes import Axes
    _noop = lambda self, *a, **k: None
    for _m in ('set_title', 'set_xlabel', 'set_ylabel', 'text'):
        setattr(Axes, _m, _noop)
import matplotlib.pyplot as plt
from matplotlib import cm, colors as mcolors
from scipy.ndimage import zoom as _zoom, grey_dilation as _gdil

D = '../data/zxdata/'
DDIR256 = D + 'train_input_3d_mixdir256_gh_hr4_256_segqc'
SEG512_PATH = '../data/zxdata/segments_qc_512.dat'
VCOL = ['#1e88e5', '#19a35b', '#ffd600']
HZ_COL = ['#2e86ff', '#19c37d', '#ffa62b', '#c355f5']
SEG_COL = ['#00e5ff', '#76ff56', '#ffe14d', '#ff7ae6', '#4fc3ff', '#c2a3ff',
           '#ffa14a', '#4dffd0']
FITC = '#ff00ff'
UXLEV = [0.3502, 0.8443]
GRIDS = [
    (256, dict(sx=D + 'sx_256x256x256.dat', frame=D + 'frame_256_256_256_hr4_gh.dat',
               XL=2, IL=78,
               iso={'input': (200, list(range(27, 248, 20))),
                    'ortho': (13, list(range(19, 240, 20)))},
               models=[('no fine-tuning', 'checkpoints/zeroshot_256.dat', False),
                       ('full fine-tuning',
                        'checkpoints/zxdata_hr4_3d_xline256_lossonly_pairdip_sr3_ep150_bs40/png_149.dat',
                        True)])),
    (512, dict(sx=D + 'sx_512x512x512.dat', frame=D + 'frame_512_hr4_gh_mod.dat',
               XL=3, IL=157,
               iso={'input': (399, list(range(55, 496, 40))),
                    'ortho': (26, list(range(39, 480, 40)))},
               models=[('no fine-tuning', 'checkpoints/zeroshot_512.dat', False),
                       ('full fine-tuning',
                        'checkpoints/zxdata_hr4_3d_xline_segqc3_pairdip_a03_lr1e4_sr3_ep150_bs40/png_149.dat',
                        True)])),
]


def detect_centers(fr, n=4):
    v = np.round(fr[fr > 0.001].astype(np.float64), 4)
    vals, cnt = np.unique(v, return_counts=True)
    order = np.argsort(cnt)[::-1]
    picked = []
    for i in order:
        if all(abs(vals[i] - p) > 0.02 for p in picked):
            picked.append(vals[i])
        if len(picked) == n:
            break
    return sorted(picked)


def pick_map(fr, c, N):
    zg = np.arange(N, dtype=np.float64)
    m = np.abs(fr - c) < 1.5e-3
    cnt = m.sum(axis=2)
    z = (m * zg[None, None, :]).sum(axis=2) / np.maximum(cnt, 1)
    z = np.where(cnt > 0, z, np.nan)
    z[(z < 2) | (z > N - 3)] = np.nan
    return z


def tau_at(PM, Z):
    N = PM.shape[2]
    z = np.clip(np.nan_to_num(Z, nan=0.0), 0, N - 1 - 1e-6)
    z0 = np.floor(z).astype(int)
    f = z - z0
    xi, yi = np.meshgrid(np.arange(PM.shape[0]), np.arange(PM.shape[1]), indexing='ij')
    pv = PM[xi, yi, z0] * (1 - f) + PM[xi, yi, np.minimum(z0 + 1, N - 1)] * f
    pv[~np.isfinite(Z)] = np.nan
    return pv


def iso_sec(pm, lv):
    nz, nt = pm.shape
    below = (pm < lv).sum(axis=0)
    z1 = np.clip(below, 1, nz - 1)
    p0 = pm[z1 - 1, np.arange(nt)]
    p1 = pm[z1, np.arange(nt)]
    frac = np.where(p1 > p0, (lv - p0) / np.maximum(p1 - p0, 1e-12), 0.0)
    zp = z1 - 1 + np.clip(frac, 0, 1)
    bad = (lv < pm[0]) | (lv > pm[-1])
    zp = zp.astype(np.float64)
    zp[bad] = np.nan
    return zp


def load_hr1_map(N):
    hv = np.fromfile(D + 'horizon_valid1_new.dat', np.float32).reshape(256, 256)
    if N == 512:
        hv = _zoom(hv, 2.0, order=1)[:N, :N] * 2.0
    return hv.astype(np.float64)


def ux_maps(N):
    UX = np.fromfile(D + 'ux_gh_256x256x256.dat', np.float32).reshape(256, 256, 256)
    PM = np.maximum.accumulate(UX.astype(np.float64), axis=2)
    maps = []
    for lv in UXLEV:
        zm = np.full((256, 256), np.nan)
        for i in range(256):
            zm[i] = iso_sec(PM[i].T, lv)
        if N == 512:
            zm = _zoom(np.nan_to_num(zm, nan=-1), 2.0, order=1)[:N, :N] * 2.0
            zm[zm < 0] = np.nan
        maps.append(zm)
    return maps


def seg_polylines(seg_zx, N, min_len):
    lines = []
    for sid in np.unique(seg_zx):
        if sid <= 0:
            continue
        zs, xs = np.nonzero(seg_zx == sid)
        cnt = np.bincount(xs, minlength=N)
        if (cnt > 0).sum() < min_len:
            continue
        zsum = np.bincount(xs, weights=zs, minlength=N)
        lines.append(np.where(cnt > 0, zsum / np.maximum(cnt, 1), np.nan))
    return lines


def seg_sections(N, g):
    """segments on the two displayed sections, as polylines."""
    if N == 256:
        d = np.load(f'{DDIR256}/{g["XL"]}.npy', allow_pickle=True).item()
        seg_in = d['segments'][0]                       # (z, x)
        seg_or = np.zeros((N, N), np.float32)
        for j in range(N):
            dj = np.load(f'{DDIR256}/{j}.npy', allow_pickle=True).item()
            seg_or[:, j] = dj['segments'][0][:, g['IL']]
        min_len = 50
    else:
        SEG = np.fromfile(SEG512_PATH, np.float32).reshape(N, N, N)
        seg_in = SEG[:, g['XL'], :].T
        seg_or = SEG[g['IL'], :, :].T
        min_len = 100
    return {'input': seg_polylines(seg_in, N, min_len),
            'ortho': seg_polylines(seg_or, N, min_len)}


def dip_sections(N, g):
    """in-plane PWD dip component on the two displayed sections (256-native,
    2-D upsampled for 512; slope values are scale-invariant)."""
    pi = np.load(D + 'pwd_dipi_256.npy').astype(np.float64)   # dz/dx
    px = np.load(D + 'pwd_dipx_256.npy').astype(np.float64)   # dz/dy
    xl0, il0 = (g['XL'], g['IL']) if N == 256 else (g['XL'] // 2, g['IL'] // 2)
    d_in = pi[:, xl0, :].T                                    # (z, x)
    d_or = px[il0, :, :].T                                    # (z, y)
    if N == 512:
        d_in = _zoom(d_in, 2.0, order=1)[:N, :N]
        d_or = _zoom(d_or, 2.0, order=1)[:N, :N]
    return {'input': d_in, 'ortho': d_or}


def depth_calibrate(v_zx):
    m = np.maximum.accumulate(np.nanmean(np.maximum.accumulate(v_zx, axis=0), axis=1))
    zt = np.linspace(0, 1, len(m))
    return np.interp(v_zx, m, zt)


try:
    import cigvis
    RGT_CMAP = cigvis.colormap.get_cmap_from_str('stratum')
except Exception:
    RGT_CMAP = 'jet'

ROWS = []
print('=== preparing rows ===', flush=True)
for N, g in GRIDS:
    sx = np.fromfile(g['sx'], np.float32).reshape(N, N, N)
    sx = (sx - sx.mean()) / sx.std()
    FR = np.fromfile(g['frame'], np.float32).reshape(N, N, N)
    cents = detect_centers(FR)
    print(f'[{N}] frame centers: {cents}', flush=True)
    IN_MAPS = [pick_map(FR, c, N) for c in cents]
    VA_MAPS = [load_hr1_map(N)] + ux_maps(N)
    segs = seg_sections(N, g)
    dips = dip_sections(N, g)
    for mname, mpath, show_seg in g['models']:
        vol = np.fromfile(mpath, np.float32).reshape(N, N, N)
        PM = np.maximum.accumulate(vol.astype(np.float64), axis=2)
        levs_in, levs_va = [], []
        mae_in_vol, mae_va_vol = [], []
        for maps, levs, maes in ((IN_MAPS, levs_in, mae_in_vol),
                                 (VA_MAPS, levs_va, mae_va_vol)):
            for Z in maps:
                pv = tau_at(PM, Z)
                lv = np.nanmedian(pv)
                levs.append(lv)
                errs = []
                for j in range(N):
                    zp = iso_sec(PM[:, j, :].T, lv)
                    d = np.abs(zp - Z[:, j])
                    d = d[np.isfinite(d)]
                    if d.size > N * 0.2:
                        errs.append(d.mean())
                maes.append(np.mean(errs))
        print(f'[{N}] {mname:16s} volume MAE  input {np.mean(mae_in_vol):.2f}'
              f'  valid {np.mean(mae_va_vol):.2f}', flush=True)
        ROWS.append(dict(N=N, g=g, sx=sx, FR=FR, IN_MAPS=IN_MAPS, VA_MAPS=VA_MAPS,
                         cents=cents, vol=vol, segs=segs, dips=dips,
                         show_seg=show_seg, levs_in=levs_in, levs_va=levs_va,
                         label=('$%d^3$, %s' % (N, mname))))


ZOOMX = {}


def draw(figname, kind):
    fig, axes = plt.subplots(4, 5, figsize=(5.0 * 5, 4.6 * 4))
    cmap = cm.get_cmap('jet')
    for r, R in enumerate(ROWS):
        N, g = R['N'], R['g']
        zg = np.arange(N, dtype=np.float64)
        xg = np.arange(N)
        if kind == 'input':
            j = g['XL']
            sec = lambda v: v[:, j, :].T
            tru = lambda Z: Z[:, j]
        else:
            i = g['IL']
            sec = lambda v: v[i, :, :].T
            tru = lambda Z: Z[i, :]
        s = sec(R['sx'])
        pr = sec(R['vol']).astype(np.float64)
        pm = np.maximum.accumulate(pr, axis=0)
        sc = N / 512.0                      # geometry scale for tick marks

        # ---- col 1: inputs
        ax = axes[r, 0]
        ax.imshow(s, cmap='gray', vmin=-2, vmax=2, aspect='auto', interpolation='bilinear')
        if R['show_seg']:
            for si, zm in enumerate(R['segs'][kind]):
                ax.plot(xg, zm, color='#111111', lw=4.6, alpha=0.75)
                ax.plot(xg, zm, color=SEG_COL[(si * 3) % len(SEG_COL)], lw=3.0, alpha=1.0)
            dsec = R['dips'][kind]
            L = 11.0 * sc
            step = max(int(46 * sc), 12)
            for xq in range(step // 2, N - step // 3, step):
                for zq in range(step // 2, N - step // 3, step):
                    d = float(dsec[zq, xq])
                    v = np.array([1.0, d]); v /= np.linalg.norm(v)
                    ax.plot([xq - L * v[0], xq + L * v[0]], [zq - L * v[1], zq + L * v[1]],
                            color='k', lw=4.2, alpha=0.85, solid_capstyle='round')
                    ax.plot([xq - L * v[0], xq + L * v[0]], [zq - L * v[1], zq + L * v[1]],
                            color='#00ffe1', lw=2.6, alpha=1.0, solid_capstyle='round')
        for Z, hc in zip(R['IN_MAPS'], HZ_COL):
            ax.plot(xg, tru(Z), color='k', lw=7.2, alpha=0.9)
            ax.plot(xg, tru(Z), color=hc, lw=4.8, alpha=1.0)
        ax.set_ylabel(R['label'], fontsize=14)
        if r == 0:
            ax.set_title('inputs: horizons (+ segments, dips)', fontsize=14)

        # ---- col 2: predicted RGT
        ax = axes[r, 1]
        ax.imshow(depth_calibrate(pr), cmap=RGT_CMAP, vmin=0, vmax=1,
                  aspect='auto', interpolation='bilinear')
        if r == 0:
            ax.set_title('predicted RGT', fontsize=14)

        # ---- col 3: iso-lines
        ax = axes[r, 2]
        ax.imshow(s, cmap='gray', vmin=-2, vmax=2, aspect='auto', interpolation='bilinear')
        iso_tr, iso_depths = g['iso'][kind]
        seed_col = pm[:, iso_tr]
        dnorm = mcolors.Normalize(vmin=min(iso_depths), vmax=max(iso_depths))
        for sd in iso_depths:
            lv = np.interp(sd, zg, seed_col)
            zp = iso_sec(pm, lv)
            ax.plot(xg, zp, color=cmap(dnorm(sd)), lw=4.4, alpha=0.8)
        if r == 0:
            ax.set_title('prediction (RGT iso-lines)', fontsize=14)

        # ---- col 4: fit to input horizons
        ax = axes[r, 3]
        ax.imshow(s, cmap='gray', vmin=-2, vmax=2, aspect='auto', interpolation='bilinear')
        fat = max(int(round(11 * sc)), 5)
        fr2 = _gdil(sec(R['FR']), size=(fat, 1)).copy()
        fr2[fr2 == 0] = np.nan
        ax.imshow(fr2, interpolation='nearest', cmap='jet', vmin=0, vmax=1, aspect='auto')
        errs_in = []
        for gi, (Z, lv) in enumerate(zip(R['IN_MAPS'], R['levs_in'])):
            zp = iso_sec(pm, lv)
            ax.plot(xg, zp, ls=':', color=('black' if gi == 3 else FITC), lw=3.6)
            d = np.abs(zp - tru(Z)); d = d[np.isfinite(d)]
            if d.size > N * 0.1:
                errs_in.append(d.mean())
        ax.set_xlabel('input-horizon MAE = %.2f px' % np.mean(errs_in), fontsize=11)
        if r == 0:
            ax.set_title('fit to input horizons', fontsize=14)

        # ---- col 5: validation fit + zoom inset on deep horizon
        ax = axes[r, 4]
        ax.imshow(s, cmap='gray', vmin=-2, vmax=2, aspect='auto', interpolation='bilinear')
        errs_va = []
        zl_deep = zp_deep = None
        for vi, (Z, lv, colr) in enumerate(zip(R['VA_MAPS'], R['levs_va'], VCOL)):
            zt = tru(Z)
            ax.plot(xg, zt, color=colr, lw=5.6, alpha=0.95)
            zp = iso_sec(pm, lv)
            ax.plot(xg, zp, 'r:', lw=3.8)
            d = np.abs(zp - zt); d = d[np.isfinite(d)]
            if d.size > N * 0.1:
                errs_va.append(d.mean())
            if vi == 2:
                zl_deep, zp_deep = zt, zp
        ax.set_xlabel('validation-horizon MAE = %.2f px' % np.mean(errs_va), fontsize=11)
        # zoom window: chosen on the no-FT row of this grid where the deep-
        # horizon error is largest; shared by both rows of the grid
        W = int(0.34 * N)
        if not R['show_seg']:
            err = np.abs(zp_deep - zl_deep)
            best, bx = -1.0, 0
            for x0 in range(0, N - W, max(N // 64, 4)):
                e = err[x0:x0 + W]
                e = e[np.isfinite(e)]
                if e.size > W * 0.5 and e.mean() > best:
                    best, bx = e.mean(), x0
            ZOOMX[(N, kind)] = bx
        X0 = ZOOMX[(N, kind)]
        X1 = X0 + W
        ZH = int(40 * sc) + 12
        zc = np.nanmean(zl_deep[X0:X1])
        ax.add_patch(plt.Rectangle((X0, max(zc - ZH, 0)), X1 - X0,
                                   min(zc + ZH, N - 1) - max(zc - ZH, 0),
                                   fill=False, ec='k', lw=2.0))
        inset_pos = [0.55, 0.18, 0.42, 0.36] if X0 < N // 2 else [0.03, 0.18, 0.42, 0.36]
        axi = ax.inset_axes(inset_pos)
        axi.imshow(s, cmap='gray', vmin=-2, vmax=2, aspect='auto', interpolation='bilinear')
        axi.plot(xg, zl_deep, color=VCOL[2], lw=6.5, alpha=0.95)
        axi.plot(xg, zp_deep, 'r:', lw=4.2)
        zlo = min(zc + ZH, N - 1); zhi = max(zc - ZH, 0)
        axi.set_xlim(X0, X1); axi.set_ylim(zlo, zhi)
        axi.set_xticks([]); axi.set_yticks([])
        for sp in axi.spines.values():
            sp.set_linewidth(2.0)
        rw = np.abs(zp_deep - zl_deep)[X0:X1]; rw = rw[np.isfinite(rw)]
        axi.text(0.04, 0.07, '%.1f px' % np.mean(rw), transform=axi.transAxes,
                 fontsize=12, color='black',
                 bbox=dict(fc='white', alpha=0.85, ec='none', pad=2))
        if r == 0:
            ax.set_title('validation-horizon fit', fontsize=14)

        for c in range(5):
            axes[r, c].set_xlim(0, N); axes[r, c].set_ylim(N, 0)
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])

    if kind == 'ortho':
        ax0 = axes[0, 0]
        ax0.annotate('', xy=(0.98, 1.12), xytext=(0.55, 1.12),
                     xycoords='axes fraction',
                     arrowprops=dict(arrowstyle='-|>', lw=2.4, color='k'))
        ax0.text(0.765, 1.17, 'Section-stacking direction', ha='center',
                 transform=ax0.transAxes, fontsize=13)
    fig.tight_layout()
    fig.savefig(figname, dpi=120, facecolor='white', bbox_inches='tight')
    print('saved', figname, flush=True)


_sfx = '_notext' if NOTEXT else ''
draw(f'qc_overlays/zx_sizepair_inputsec{_sfx}.png', 'input')
draw(f'qc_overlays/zx_sizepair_orthosec{_sfx}.png', 'ortho')
