# Data format

## Coordinate convention

Volumes use the logical layout `(n_section, n_trace, n_depth)`. Depth sample
indices increase downward. The section axis is the direction in which the 2-D
network is applied, and `n_section` is the section-stacking direction.

All dense floating-point arrays are stored as `float32`. Segment labels are
non-negative integers; zero denotes an unlabeled sample.

## Per-section training dictionary

The training dataset contains one file per section, named `0.npy`, `1.npy`,
and so forth. Each file is a dictionary saved with `numpy.save`:

| Key | Shape | Meaning |
|---|---|---|
| `seis` | `(1, nz, nx)` | normalized seismic amplitudes |
| `frame` | `(1, nz, nx)` | interpreted-horizon input channel; zero elsewhere |
| `mx_single` | `(nh, nz, nx)` | one binary mask per interpreted horizon |
| `segments` | `(1, nz, nx)` | globally or section-wise labeled horizon segments |
| `mask_valid` | `(1, 1, nz, nx)` | samples eligible for loss evaluation |

The network input is the three-channel tensor `[10 * frame, seis, seis]`.
The factor of ten follows the normalization used when pretraining the released
RGT network.

`mx_single` stores masks rather than target RGT values. The interpreted-horizon
loss computes the predicted mean on a horizon during each update and penalizes
deviations from that mean.

## Cross-section dip volume

The local-dip file is a NumPy array with shape `(n_section, nz, nx)`. For
adjacent sections `i` and `i+1`, a positive value `delta_i(z, x)` means that
the same reflection is deeper in section `i+1`:

```text
tau_i(z, x) ~= tau_{i+1}(z + delta_i(z, x), x)
```

Dip is measured in depth samples per section spacing. The released configs use
adjacent sections (`pair_gap = 1`). Values sampled outside the depth range are
excluded from the structure loss.

## Predicted RGT volume

The evaluation program reads a raw `float32` file with layout
`(n1, n2, n_depth)`. Supply the dimensions explicitly through `--shape`.
Horizon frames use the same layout.
