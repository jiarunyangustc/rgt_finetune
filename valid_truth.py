"""统一验证层位真值加载。
hr1(浅层): 用独立拾取的 horizon_valid1_new.dat (比 ux_gh 派生等值面更干净, drift 更可信)。
  文件: ../data/zxdata/horizon_valid1_new.dat, 256x256, 轴序 [inline, xline], 深度单位=256网格。
  512 用: 面 ×2 上采样 + 深度值 ×2。
返回 ZT[j] = xline切片 j 上的层位深度(over inline), 长度 N。
"""
import numpy as np
from scipy.ndimage import zoom as _zoom

HR1_PATH = '../data/zxdata/horizon_valid1_new.dat'
MH_PATH = '../data/zxdata/mh_valid_final.dat'                   # 中层拾取真值 (z≈104-144, 大空隙中部)


def _load_map(path, N):
    hv = np.fromfile(path, np.float32).reshape(256, 256)        # [inline, xline], depth in 256
    if N == 512:
        hv = _zoom(hv, 2.0, order=1)[:N, :N] * 2.0              # 面×2, 深度×2
    elif N != 256:
        raise ValueError(f'unsupported N={N}')
    return [hv[:, j].astype(np.float64) for j in range(N)]      # ZT[j] over inline


def load_hr1(N):
    return _load_map(HR1_PATH, N)


def load_mh(N):
    return _load_map(MH_PATH, N)
