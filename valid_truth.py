"""Load validation-horizon depth maps used by the paper experiments.

The shallow horizon is an independent pick stored on the native 256 grid with
axis order ``[inline, crossline]``. For the 512 grid, the surface coordinates
and depth values are both scaled by two. The returned ``ZT[j]`` is the horizon
depth vector along inline for crossline section ``j``.
"""
import numpy as np
from scipy.ndimage import zoom as _zoom

HR1_PATH = '../data/zxdata/horizon_valid1_new.dat'
MH_PATH = '../data/zxdata/mh_valid_final.dat'


def _load_map(path, N):
    hv = np.fromfile(path, np.float32).reshape(256, 256)        # [inline, xline], depth in 256
    if N == 512:
        hv = _zoom(hv, 2.0, order=1)[:N, :N] * 2.0
    elif N != 256:
        raise ValueError(f'unsupported N={N}')
    return [hv[:, j].astype(np.float64) for j in range(N)]      # ZT[j] over inline


def load_hr1(N):
    return _load_map(HR1_PATH, N)


def load_mh(N):
    return _load_map(MH_PATH, N)
