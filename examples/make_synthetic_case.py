#!/usr/bin/env python3
"""Create a deterministic synthetic case following the public data contract."""

import argparse
from pathlib import Path

import numpy as np


def _depth_for_level(rgt: np.ndarray, level: float) -> np.ndarray:
    """Return nearest depth indices for one RGT level on every trace."""
    return np.abs(rgt - level).argmin(axis=-1).astype(np.float32)


def _surface_mask(depth: np.ndarray, nz: int) -> np.ndarray:
    mask = np.zeros(depth.shape + (nz,), dtype=np.float32)
    ii, jj = np.indices(depth.shape)
    mask[ii, jj, np.rint(depth).astype(int)] = 1.0
    return mask


def build_case(shape=(24, 16, 64), seed=2026):
    """Build seismic, prior, dip, and prediction arrays for a small example."""
    nsec, ntrace, nz = shape
    rng = np.random.default_rng(seed)
    sec, trace, depth = np.meshgrid(
        np.arange(nsec), np.arange(ntrace), np.arange(nz), indexing="ij"
    )

    structural_shift = (
        2.2 * np.sin(2.0 * np.pi * sec / max(nsec - 1, 1))
        + 1.4 * np.sin(2.0 * np.pi * trace / max(ntrace - 1, 1))
        + 0.035 * (sec - nsec / 2.0) * (trace - ntrace / 2.0) / ntrace
    )
    true_rgt = np.clip((depth - structural_shift) / (nz - 1), 0.0, 1.0)
    seismic = (
        np.sin(2.0 * np.pi * 11.0 * true_rgt)
        + 0.45 * np.sin(2.0 * np.pi * 19.0 * true_rgt + 0.4)
        + 0.08 * rng.standard_normal(shape)
    ).astype(np.float32)
    seismic = (seismic - seismic.mean()) / (seismic.std() + 1.0e-8)

    input_levels = (0.25, 0.72)
    validation_level = 0.50
    input_depths = [_depth_for_level(true_rgt, lv) for lv in input_levels]
    validation_depth = _depth_for_level(true_rgt, validation_level)

    frame = np.zeros(shape, dtype=np.float32)
    masks = []
    for lv, zmap in zip(input_levels, input_depths):
        mask = _surface_mask(zmap, nz)
        frame[mask > 0] = lv
        masks.append(mask)

    segments = np.zeros(shape, dtype=np.int32)
    label = 1
    for level in np.linspace(0.12, 0.90, 12):
        zmap = _depth_for_level(true_rgt, float(level))
        for start in range(0, ntrace, 4):
            stop = min(start + 3, ntrace)
            for i in range(nsec):
                for j in range(start, stop):
                    segments[i, j, int(zmap[i, j])] = label
            label += 1

    pair_dip = np.zeros((nsec, nz, ntrace), dtype=np.float32)
    for i in range(nsec - 1):
        shift = structural_shift[i + 1] - structural_shift[i]
        pair_dip[i] = shift.T

    section_bias = 0.055 * np.sin(2.0 * np.pi * sec / 3.0)
    prediction_direct = np.clip(
        true_rgt + section_bias + 0.012 * rng.standard_normal(shape), 0.0, 1.0
    ).astype(np.float32)
    prediction_adapted = np.clip(
        true_rgt + 0.004 * rng.standard_normal(shape), 0.0, 1.0
    ).astype(np.float32)

    return {
        "seismic": seismic.astype(np.float32),
        "true_rgt": true_rgt.astype(np.float32),
        "frame": frame,
        "masks": np.stack(masks, axis=1),
        "segments": segments,
        "pair_dip": pair_dip,
        "validation_depth": validation_depth,
        "prediction_direct": prediction_direct,
        "prediction_adapted": prediction_adapted,
        "levels": input_levels,
    }


def write_case(output: Path, shape=(24, 16, 64), seed=2026) -> None:
    case = build_case(shape=shape, seed=seed)
    output.mkdir(parents=True, exist_ok=True)
    sections = output / "sections"
    sections.mkdir(exist_ok=True)

    np.save(output / "seismic.npy", case["seismic"])
    np.save(output / "true_rgt.npy", case["true_rgt"])
    np.save(output / "segments.npy", case["segments"])
    np.save(output / "pair_dip.npy", case["pair_dip"])
    np.save(output / "validation_horizon.npy", case["validation_depth"])
    case["frame"].tofile(output / "horizon_frame.dat")
    case["prediction_direct"].tofile(output / "prediction_direct.dat")
    case["prediction_adapted"].tofile(output / "prediction_adapted.dat")

    nsec = shape[0]
    for i in range(nsec):
        sample = {
            "seis": case["seismic"][i].T[None].astype(np.float32),
            "frame": case["frame"][i].T[None].astype(np.float32),
            "mx_single": case["masks"][i].transpose(0, 2, 1).astype(np.float32),
            "segments": case["segments"][i].T[None].astype(np.float32),
            "mask_valid": np.ones((1, 1, shape[2], shape[1]), np.float32),
        }
        np.save(sections / f"{i}.npy", sample)

    print(f"Synthetic case written to: {output.resolve()}")
    print(f"Volume shape: {shape}; section files: {nsec}")
    print("Input horizon levels: " + ", ".join(str(v) for v in case["levels"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("examples/_synthetic_case"))
    parser.add_argument("--shape", default="24,16,64", help="n_section,n_trace,n_depth")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    shape = tuple(int(v) for v in args.shape.split(","))
    if len(shape) != 3 or min(shape) < 4:
        raise ValueError("--shape must contain three integers, each at least four")
    write_case(args.output, shape=shape, seed=args.seed)


if __name__ == "__main__":
    main()
