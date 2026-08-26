#!/usr/bin/env python3
import argparse
import heapq
import json
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np


DIRECTIONS = ['N', 'S', 'E', 'W']
REV_DIR = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}


def parse_thr_list(text):
    rounds = []
    for item in text.split(';'):
        vals = tuple(float(v) for v in item.split(','))
        if len(vals) != 3:
            raise argparse.ArgumentTypeError('each round must be amp,wave,norm')
        rounds.append(vals)
    return rounds


def waveform_corr_one(wave1, wave2):
    w1 = (wave1 - np.mean(wave1)) / (np.std(wave1) + 1e-16)
    w2 = (wave2 - np.mean(wave2)) / (np.std(wave2) + 1e-16)
    correlation = np.dot(w1, w2) / (np.linalg.norm(w1) * np.linalg.norm(w2) + 1e-16)
    return np.clip(correlation, 0, 1)


def waveform_corr(wave1, wave2, window_sizes=(20, 40, 60, 80)):
    scores = []
    for w in window_sizes:
        start = int((wave1.shape[0] - 1) / 2) - w
        w1 = wave1[start: wave1.shape[0] - start]
        w2 = wave2[start: wave2.shape[0] - start]
        scores.append(waveform_corr_one(w1, w2))
    return float(np.mean(scores))



def make_wave_corr_cache(candidate_waves, original_indices, half_width=80, max_cache=200000):
    cache = OrderedDict()

    def norm_wave(sorted_idx):
        original_idx = int(original_indices[sorted_idx])
        cached = cache.get(original_idx)
        if cached is not None:
            cache.move_to_end(original_idx)
            return cached
        wave = candidate_waves[original_idx]
        center = (wave.shape[0] - 1) // 2
        start = max(0, center - half_width)
        end = min(wave.shape[0], center + half_width + 1)
        w = np.asarray(wave[start:end], dtype=np.float32)
        w = w - float(np.mean(w))
        denom = float(np.linalg.norm(w)) + 1e-16
        w = w / denom
        cache[original_idx] = w
        if len(cache) > max_cache:
            cache.popitem(last=False)
        return w

    def corr(sorted_idx1, sorted_idx2):
        w1 = norm_wave(sorted_idx1)
        w2 = norm_wave(sorted_idx2)
        return float(np.clip(np.dot(w1, w2), 0, 1))

    def corr_wave(sorted_idx1, wave2):
        w1 = norm_wave(sorted_idx1)
        center = (wave2.shape[0] - 1) // 2
        start = max(0, center - half_width)
        end = min(wave2.shape[0], center + half_width + 1)
        w2 = np.asarray(wave2[start:end], dtype=np.float32)
        w2 = w2 - float(np.mean(w2))
        w2 = w2 / (float(np.linalg.norm(w2)) + 1e-16)
        return float(np.clip(np.dot(w1, w2), 0, 1))

    return corr, corr_wave


def amplitude_corr(a1, a2, sx_min, sx_max, allow_range=0.01):
    diff = abs(float(a1) - float(a2))
    threshold = (sx_max - sx_min) * allow_range
    score = 1 - 0.1 * (diff / threshold)
    return float(np.clip(score, 0, 1))


def norm_corr(norm1, norm2):
    norm1 = np.asarray(norm1)
    norm2 = np.asarray(norm2)
    correlation = np.dot(norm1, norm2) / (np.linalg.norm(norm1) * np.linalg.norm(norm2) + 1e-8)
    return float(np.clip(correlation, 0, 1))


def calc_score(amp_score, wave_score, norm_score, amp_thr, wave_thr, norm_thr):
    if amp_score < amp_thr or wave_score < wave_thr or norm_score < norm_thr:
        return 0.0
    return float(amp_score * wave_score * norm_score)


class OnlineStats:
    def __init__(self, init_values):
        self.n = len(init_values)
        self.mean = float(np.mean(init_values))
        self.M2 = float(np.sum((init_values - self.mean) ** 2))

    def update(self, new_value):
        self.n += 1
        delta = float(new_value) - self.mean
        self.mean += delta / self.n
        delta2 = float(new_value) - self.mean
        self.M2 += delta * delta2

    @property
    def std(self):
        return float(np.sqrt(self.M2 / (self.n - 1 + 1e-8)))


def global_amp_wave_constraint(new_amp, new_wave, amp_stats, mean_wave, z_thr=2.5, corr_thr=0.5):
    z = (float(new_amp) - amp_stats.mean) / (amp_stats.std + 1e-8)
    if abs(z) >= z_thr:
        return False
    corr = waveform_corr(new_wave, mean_wave)
    return corr >= corr_thr


DIR_STEP = {'N': (0, -1), 'S': (0, 1), 'E': (1, 0), 'W': (-1, 0)}


def get_neighbor_indices(coord, direction, candidate_dict, shape, width=2, dz_center=0):
    """dz_center: 倾角预测的 z 偏移中心（默认 0 = 原始行为，窗口对称于当前 z）"""
    x, y, z = coord
    dx, dy = DIR_STEP[direction]
    neighbors = []
    for dz in range(dz_center - width, dz_center + width + 1):
        nx, ny, nz = x + dx, y + dy, z + dz
        if 0 <= nx < shape[0] and 0 <= ny < shape[1] and 0 <= nz < shape[2]:
            key = (nx, ny, nz)
            if key in candidate_dict:
                neighbors.append(key)
    return neighbors


def predict_dz(normal, direction, axis_map):
    """由局部法向预测沿 direction 走一步的 z 偏移（四舍五入到整数采样点）。

    平面法向 n 满足 n·(dx,dy,dz)=0 => dz = -(n_x*dx + n_y*dy) / n_z。
    axis_map: (ix, iy, iz) —— normal 向量中对应体数据 x/y/z 轴的分量下标。
    """
    dx, dy = DIR_STEP[direction]
    ix, iy, iz = axis_map
    nz = float(normal[iz])
    if abs(nz) < 0.3:   # 近垂直构造（法向躺平），预测不可靠，退回不引导
        return 0
    dz = -(float(normal[ix]) * dx + float(normal[iy]) * dy) / nz
    return int(round(np.clip(dz, -8, 8)))


def infer_normal_axis_map(candidate_norms):
    """推断 normal 分量与体数据轴的对应：|均值| 最大的分量是深度(z)方向。"""
    mean_abs = np.mean(np.abs(candidate_norms), axis=0)
    iz = int(np.argmax(mean_abs))
    rest = [k for k in range(3) if k != iz]
    axis_map = (rest[0], rest[1], iz)
    print('normal 分量 |mean|=%s -> 深度分量 idx=%d, axis_map(x,y,z)=%s' % (
        np.round(mean_abs, 3).tolist(), iz, axis_map), flush=True)
    return axis_map


def prune_surface_points(surface_points, resid_thr, xy_radius=4, min_neighbors=5):
    """面级清洗：每点与其 xy 邻域(切比雪夫半径)内点的中位 z 比较，残差超阈值剔除。"""
    xy2z = {}
    for (x, y, z) in surface_points:
        xy2z[(x, y)] = z
    kept = []
    removed = 0
    for (x, y, z) in surface_points:
        zs = []
        for ddx in range(-xy_radius, xy_radius + 1):
            for ddy in range(-xy_radius, xy_radius + 1):
                if ddx == 0 and ddy == 0:
                    continue
                zn = xy2z.get((x + ddx, y + ddy))
                if zn is not None:
                    zs.append(zn)
        if len(zs) < min_neighbors:
            kept.append((x, y, z))
            continue
        if abs(z - float(np.median(zs))) > resid_thr:
            removed += 1
        else:
            kept.append((x, y, z))
    return kept, removed


def crosses_fault(current, neighbor, fault_barrier):
    if fault_barrier is None:
        return False
    x0, y0, z0 = current
    x1, y1, z1 = neighbor
    steps = max(abs(x1 - x0), abs(y1 - y0), abs(z1 - z0)) + 1
    xs = np.rint(np.linspace(x0, x1, steps)).astype(np.int64)
    ys = np.rint(np.linspace(y0, y1, steps)).astype(np.int64)
    zs = np.rint(np.linspace(z0, z1, steps)).astype(np.int64)
    return bool(np.any(fault_barrier[xs, ys, zs] > 0))


def build_segment_volume(surfaces, shape):
    segments = np.zeros(shape, dtype=np.int32)
    for sid in range(1, len(surfaces) + 1):
        pts = np.asarray(surfaces[str(sid)], dtype=np.int32)
        if pts.size == 0:
            continue
        segments[pts[:, 0], pts[:, 1], pts[:, 2]] = sid
    return segments


def grow_surfaces_multi_rounds(
    sx,
    candidate_coords,
    candidate_li,
    candidate_amps,
    candidate_norms,
    candidate_waves,
    original_indices,
    thr_list,
    width=2,
    max_surface_size=100000,
    min_surface_size=5000,
    li_thr=0.1,
    global_amp_z_thr=3.0,
    global_wave_corr_thr=0.65,
    global_constraint_min_points=500,
    fault_barrier=None,
    max_seeds=None,
    dip_guided=False,
    dip_window=1,
    prune_resid=0.0,
    prune_radius=4,
):
    shape = sx.shape
    sx_min, sx_max = float(sx.min()), float(sx.max())
    sort_idx = np.argsort(candidate_li)[::-1]
    if max_seeds is not None:
        sort_idx = sort_idx[:max_seeds]

    sorted_coords = candidate_coords[sort_idx]
    sorted_li = candidate_li[sort_idx]
    sorted_amps = candidate_amps[sort_idx]
    sorted_norms = candidate_norms[sort_idx]
    sorted_original_indices = original_indices[sort_idx]

    candidate_dict = {tuple(coord): idx for idx, coord in enumerate(sorted_coords)}
    used = set()
    surfaces = {}
    surfaces_points = {}
    surface_id = 1

    axis_map = None
    if dip_guided:
        axis_map = infer_normal_axis_map(sorted_norms)
        print('倾角引导开启: 搜索窗以法向预测 z 为中心, 半宽=%d' % dip_window, flush=True)
    if prune_resid > 0:
        print('面级清洗开启: xy 半径=%d 中位 z 残差 > %.1f 采样点的点剔除' % (prune_radius, prune_resid), flush=True)

    def wave_at(sorted_idx):
        return candidate_waves[int(sorted_original_indices[sorted_idx])]

    wave_corr_idx, wave_corr_to_wave = make_wave_corr_cache(
        candidate_waves, sorted_original_indices, half_width=80, max_cache=200000
    )

    for seed_idx, seed in enumerate(sorted_coords):
        if seed_idx % 10000 == 0:
            print('[%d / %d] surfaces=%d' % (seed_idx, sorted_coords.shape[0], surface_id - 1), flush=True)
        seed_key = tuple(int(v) for v in seed)
        if seed_key in used:
            continue
        if fault_barrier is not None and fault_barrier[seed_key] > 0:
            continue

        surface_points = [seed_key]
        used.add(seed_key)
        xy_set = {(seed_key[0], seed_key[1])}
        point_dir_flag = {seed_key: {d: False for d in DIRECTIONS}}
        round_points = []
        seed_rank = candidate_dict[seed_key]
        amp_stats = OnlineStats(np.array([sorted_amps[seed_rank]], dtype=np.float64))
        mean_wave = np.copy(wave_at(seed_rank)).astype(np.float64)
        mean_wave_count = 1

        for round_idx, (amp_thr, wave_thr, norm_thr) in enumerate(thr_list):
            if round_idx == 0:
                frontier = [(-float(sorted_li[seed_rank]), seed_key)]
            else:
                incomplete_points = [pt for pt, flag in point_dir_flag.items() if not all(flag.values())]
                frontier = [(-float(sorted_li[candidate_dict[pt]]), pt) for pt in incomplete_points]
            heapq.heapify(frontier)

            while frontier:
                _, current = heapq.heappop(frontier)
                cur_idx = candidate_dict[current]
                if sorted_li[cur_idx] < li_thr:
                    continue
                if current not in point_dir_flag:
                    point_dir_flag[current] = {d: False for d in DIRECTIONS}

                for direction in DIRECTIONS:
                    if point_dir_flag[current][direction]:
                        continue
                    if dip_guided:
                        dzc = predict_dz(sorted_norms[cur_idx], direction, axis_map)
                        neighbors = get_neighbor_indices(current, direction, candidate_dict, shape, dip_window, dzc)
                    else:
                        neighbors = get_neighbor_indices(current, direction, candidate_dict, shape, width)
                    best_score = amp_thr * wave_thr * norm_thr
                    best_neighbor = None

                    for nb in neighbors:
                        if nb in used:
                            continue
                        if (nb[0], nb[1]) in xy_set:
                            continue
                        if crosses_fault(current, nb, fault_barrier):
                            continue
                        nb_idx = candidate_dict[nb]
                        if sorted_li[nb_idx] < li_thr:
                            continue

                        if len(surface_points) >= global_constraint_min_points:
                            z = (float(sorted_amps[nb_idx]) - amp_stats.mean) / (amp_stats.std + 1e-8)
                            if abs(z) >= global_amp_z_thr:
                                continue
                            if wave_corr_to_wave(nb_idx, mean_wave) < global_wave_corr_thr:
                                continue

                        amp_score = amplitude_corr(sorted_amps[cur_idx], sorted_amps[nb_idx], sx_min, sx_max)
                        if amp_score <= amp_thr:
                            continue
                        wave_score = wave_corr_idx(cur_idx, nb_idx)
                        if wave_score <= wave_thr:
                            continue
                        norm_score = norm_corr(sorted_norms[cur_idx], sorted_norms[nb_idx])
                        if norm_score <= norm_thr:
                            continue

                        score = calc_score(amp_score, wave_score, norm_score, amp_thr, wave_thr, norm_thr)
                        if score > best_score:
                            best_score = score
                            best_neighbor = nb

                    if best_score > 0 and best_neighbor is not None:
                        rev_dir = REV_DIR[direction]
                        nb_idx = candidate_dict[best_neighbor]
                        if dip_guided:
                            rev_dzc = predict_dz(sorted_norms[nb_idx], rev_dir, axis_map)
                            rev_neighbors = get_neighbor_indices(best_neighbor, rev_dir, candidate_dict, shape, dip_window, rev_dzc)
                        else:
                            rev_neighbors = get_neighbor_indices(best_neighbor, rev_dir, candidate_dict, shape, width)
                        best_rev_score = amp_thr * wave_thr * norm_thr
                        best_rev_neighbor = None

                        if len(rev_neighbors) == 1:
                            if not crosses_fault(best_neighbor, rev_neighbors[0], fault_barrier):
                                best_rev_neighbor = rev_neighbors[0]
                        else:
                            for rev_nb in rev_neighbors:
                                if crosses_fault(best_neighbor, rev_nb, fault_barrier):
                                    continue
                                rev_nb_idx = candidate_dict[rev_nb]
                                if sorted_li[rev_nb_idx] < li_thr:
                                    continue
                                amp_score = amplitude_corr(sorted_amps[nb_idx], sorted_amps[rev_nb_idx], sx_min, sx_max)
                                if amp_score <= amp_thr:
                                    continue
                                wave_score = wave_corr_idx(nb_idx, rev_nb_idx)
                                if wave_score <= wave_thr:
                                    continue
                                norm_score = norm_corr(sorted_norms[nb_idx], sorted_norms[rev_nb_idx])
                                if norm_score <= norm_thr:
                                    continue
                                score = calc_score(amp_score, wave_score, norm_score, amp_thr, wave_thr, norm_thr)
                                if score > best_rev_score:
                                    best_rev_score = score
                                    best_rev_neighbor = rev_nb

                        if best_rev_neighbor == current:
                            amp_stats.update(sorted_amps[nb_idx])
                            mean_wave = (mean_wave * mean_wave_count + wave_at(nb_idx)) / (mean_wave_count + 1)
                            mean_wave_count += 1
                            surface_points.append(best_neighbor)
                            used.add(best_neighbor)
                            xy_set.add((best_neighbor[0], best_neighbor[1]))
                            heapq.heappush(frontier, (-float(sorted_li[nb_idx]), best_neighbor))
                            point_dir_flag[current][direction] = True
                            if best_neighbor not in point_dir_flag:
                                point_dir_flag[best_neighbor] = {d: False for d in DIRECTIONS}
                            point_dir_flag[best_neighbor][rev_dir] = True
                            if len(surface_points) >= max_surface_size:
                                frontier = []
                                break
            round_points.append(len(surface_points))
            if len(surface_points) >= max_surface_size:
                break

        if prune_resid > 0 and len(surface_points) >= min_surface_size:
            kept, removed = prune_surface_points(surface_points, prune_resid, prune_radius)
            if removed:
                kept_set = set(kept)
                for p in surface_points:
                    if p not in kept_set:
                        used.discard(p)   # 释放被剔除的点，允许其他面认领
                surface_points = kept

        if len(surface_points) >= min_surface_size:
            surfaces[str(surface_id)] = surface_points
            surfaces_points[str(surface_id)] = round_points
            surface_id += 1
    return surfaces, surfaces_points


def run_one(kind, args, data_dir, sx, li, norm, fault_barrier):
    sk_path = data_dir / ('sk_%s_512.npy' % kind)
    wave_path = data_dir / ('waveform_%s_up2_512.npy' % kind)
    print('loading %s skeleton: %s' % (kind, sk_path), flush=True)
    sk = np.load(str(sk_path), mmap_mode='r')
    candidate_coords_all = np.argwhere(sk == 1)
    original_indices_all = np.arange(candidate_coords_all.shape[0], dtype=np.int64)
    candidate_amps_all = sx[sk == 1].astype(np.float32)
    print('%s candidates before amplitude filter: %d' % (kind, candidate_coords_all.shape[0]), flush=True)

    if args.amp_percentile is not None:
        amp_thr = float(np.percentile(np.abs(sx), args.amp_percentile))
        amp_mask = np.abs(candidate_amps_all) >= amp_thr
        candidate_coords = candidate_coords_all[amp_mask]
        original_indices = original_indices_all[amp_mask]
        candidate_amps = candidate_amps_all[amp_mask]
        print('%s amplitude filter: abs(sx) >= P%.1f = %.4f, kept %d / %d' % (
            kind, args.amp_percentile, amp_thr, candidate_coords.shape[0], candidate_coords_all.shape[0]
        ), flush=True)
    else:
        candidate_coords = candidate_coords_all
        original_indices = original_indices_all
        candidate_amps = candidate_amps_all

    sk_mask = np.zeros(sk.shape, dtype=bool)
    if candidate_coords.size > 0:
        sk_mask[candidate_coords[:, 0], candidate_coords[:, 1], candidate_coords[:, 2]] = True
    candidate_li = li[sk_mask].astype(np.float32)
    candidate_norms = norm[sk_mask].astype(np.float32)
    candidate_waves = np.load(str(wave_path), mmap_mode='r')

    if candidate_waves.shape[0] != candidate_coords_all.shape[0]:
        raise ValueError('%s wave count mismatch: %s vs %s' % (kind, candidate_waves.shape, candidate_coords_all.shape))
    print('%s candidates after filters: %d' % (kind, candidate_coords.shape[0]), flush=True)

    start = time.time()
    surfaces, points = grow_surfaces_multi_rounds(
        sx=sx,
        candidate_coords=candidate_coords,
        candidate_li=candidate_li,
        candidate_amps=candidate_amps,
        candidate_norms=candidate_norms,
        candidate_waves=candidate_waves,
        original_indices=original_indices,
        thr_list=args.thr_list,
        width=args.width,
        max_surface_size=args.max_surface_size,
        min_surface_size=args.min_surface_size,
        li_thr=args.li_thr,
        global_amp_z_thr=args.global_amp_z_thr,
        global_wave_corr_thr=args.global_wave_corr_thr,
        global_constraint_min_points=args.global_constraint_min_points,
        fault_barrier=fault_barrier,
        max_seeds=args.max_seeds,
        dip_guided=args.dip_guided,
        dip_window=args.dip_window,
        prune_resid=args.prune_resid,
        prune_radius=args.prune_radius,
    )
    elapsed = time.time() - start
    prefix = '%s_faultbarrier_w%d_min%d_max%d' % (kind, args.width, args.min_surface_size, args.max_surface_size)
    surf_path = data_dir / ('surfaces_%s.npy' % prefix)
    pts_path = data_dir / ('surfaces_points_%s.npy' % prefix)
    np.save(str(surf_path), surfaces)
    np.save(str(pts_path), points)
    print('%s surfaces=%d time=%.1fs saved=%s' % (kind, len(surfaces), elapsed, surf_path), flush=True)
    return surfaces, points, surf_path


def combine_peak_trough(peak_surfaces, trough_surfaces, shape):
    segments = np.zeros(shape, dtype=np.int32)
    sid = 1
    for surfaces in (peak_surfaces, trough_surfaces):
        for local_id in range(1, len(surfaces) + 1):
            pts = np.asarray(surfaces[str(local_id)], dtype=np.int32)
            if pts.size == 0:
                continue
            segments[pts[:, 0], pts[:, 1], pts[:, 2]] = sid
            sid += 1
    return segments


def main():
    parser = argparse.ArgumentParser(description='Regenerate zxdata 512 peak/trough segments with a fault barrier.')
    parser.add_argument('--data-dir', default='../data/zxdata')
    parser.add_argument('--fault-mask', default='../data/zxdata/fault_512_perm210_mask_dil1.npy')
    parser.add_argument('--kinds', default='peak,trough', help='peak,trough or one of them')
    parser.add_argument('--thr-list', type=parse_thr_list, default=parse_thr_list('0.90,0.95,0.90;0.85,0.90,0.90;0.80,0.85,0.90'))
    parser.add_argument('--width', type=int, default=2)
    parser.add_argument('--amp-percentile', type=float, default=None, help='Keep only candidate points with abs(seismic) >= percentile(abs(seismic)).')
    parser.add_argument('--min-surface-size', type=int, default=5000)
    parser.add_argument('--max-surface-size', type=int, default=100000)
    parser.add_argument('--li-thr', type=float, default=0.1)
    parser.add_argument('--global-amp-z-thr', type=float, default=3.0)
    parser.add_argument('--global-wave-corr-thr', type=float, default=0.65)
    parser.add_argument('--global-constraint-min-points', type=int, default=500)
    parser.add_argument('--max-seeds', type=int, default=None, help='debug only: process only top-N seeds')
    parser.add_argument('--no-combine', action='store_true')
    parser.add_argument('--dip-guided', action='store_true',
                        help='用法向预测邻列 z 位置，搜索窗中心随倾角移动（推荐，抑制跳轴）')
    parser.add_argument('--dip-window', type=int, default=1,
                        help='倾角引导模式下的 z 搜索半宽（默认 1，替代 --width）')
    parser.add_argument('--prune-resid', type=float, default=0.0,
                        help='面级清洗：与 xy 邻域中位 z 的残差超过该采样点数则剔除（0=关闭，推荐 2.5）')
    parser.add_argument('--prune-radius', type=int, default=4,
                        help='面级清洗的 xy 邻域切比雪夫半径')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    sx = np.load(str(data_dir / 'sx_512x512x512.npy'), mmap_mode='r')
    li = np.load(str(data_dir / 'linearity_512.npy'), mmap_mode='r')
    norm = np.load(str(data_dir / 'norm_512.npy'), mmap_mode='r')
    fault_barrier = np.load(args.fault_mask, mmap_mode='r') if args.fault_mask else None

    print('sx=%s li=%s norm=%s fault=%s' % (sx.shape, li.shape, norm.shape, None if fault_barrier is None else fault_barrier.shape), flush=True)
    print('thr_list=%s width=%d' % (args.thr_list, args.width), flush=True)

    generated = {}
    for kind in [k.strip() for k in args.kinds.split(',') if k.strip()]:
        surfaces, points, surf_path = run_one(kind, args, data_dir, sx, li, norm, fault_barrier)
        generated[kind] = surfaces

    if not args.no_combine and 'peak' in generated and 'trough' in generated:
        segments = combine_peak_trough(generated['peak'], generated['trough'], sx.shape)
        out = data_dir / ('segment_pt_faultbarrier_w%d_min%d_max%d_512.npy' % (args.width, args.min_surface_size, args.max_surface_size))
        np.save(str(out), segments)
        print('combined segments saved=%s labels=%d' % (out, int(segments.max())), flush=True)

    meta = {
        'data_dir': str(data_dir),
        'fault_mask': args.fault_mask,
        'kinds': args.kinds,
        'thr_list': args.thr_list,
        'width': args.width,
        'amp_percentile': args.amp_percentile,
        'min_surface_size': args.min_surface_size,
        'max_surface_size': args.max_surface_size,
        'li_thr': args.li_thr,
        'global_amp_z_thr': args.global_amp_z_thr,
        'global_wave_corr_thr': args.global_wave_corr_thr,
        'global_constraint_min_points': args.global_constraint_min_points,
        'max_seeds': args.max_seeds,
        'dip_guided': args.dip_guided,
        'dip_window': args.dip_window,
        'prune_resid': args.prune_resid,
        'prune_radius': args.prune_radius,
    }
    meta_path = data_dir / ('segment_faultbarrier_w%d_params.json' % args.width)
    with open(str(meta_path), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    print('params saved=%s' % meta_path, flush=True)


if __name__ == '__main__':
    main()
