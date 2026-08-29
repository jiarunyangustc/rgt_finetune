"""Optional training-time geometric quality control for predicted RGT.

This diagnostic compares isochron geometry rather than RGT values, so it is
invariant to monotonic reparameterization. Dense reference horizons are
extracted once from a trusted RGT volume. During training, their vertical
deviation from corresponding predicted isochrons is reported by depth band.
The diagnostic does not contribute to the loss or gradient.
"""

import numpy as np

try:
    from scipy.ndimage import zoom as _zoom
except ImportError:
    _zoom = None


def _upsample2(array):
    if _zoom is not None:
        return _zoom(array, 2.0, order=1)
    return np.repeat(np.repeat(array, 2, axis=0), 2, axis=1)


def _monotonic(array):
    return np.maximum.accumulate(array, axis=0)


def _horizon_depths(volume_zx, levels):
    depth = np.arange(volume_zx.shape[0], dtype=np.float64)
    output = np.full((len(levels), volume_zx.shape[1]), np.nan)
    for trace in range(volume_zx.shape[1]):
        output[:, trace] = np.interp(
            levels, volume_zx[:, trace], depth, left=np.nan, right=np.nan
        )
    return output


def _sample_along(volume_zx, depths):
    depth = np.arange(volume_zx.shape[0], dtype=np.float64)
    output = np.full_like(depths, np.nan)
    for trace in range(volume_zx.shape[1]):
        valid = np.isfinite(depths[:, trace])
        if valid.any():
            output[valid, trace] = np.interp(
                depths[valid, trace], depth, volume_zx[:, trace]
            )
    return output


class RGTQualityControl:
    """Evaluate geometric horizon deviations during full-volume inference."""

    N_BANDS = 8

    def __init__(
        self,
        reference_path,
        data_shape=(512, 512, 512),
        reference_shape=(256, 256, 256),
        slice_step=32,
        levels_n=40,
        pred_local="xline",
    ):
        n1, n2, n3 = data_shape
        self.n1 = n1
        self.band = n1 // self.N_BANDS
        self.pred_local = pred_local
        reference = np.fromfile(reference_path, dtype=np.float32).reshape(reference_shape)
        scale = n2 // reference_shape[1]
        n_sections = n2 if pred_local == "xline" else n3
        self.slices = list(range(slice_step // 2, n_sections, slice_step))
        quantiles = np.linspace(3, 97, levels_n)
        self.ref_horizons = {}
        for section in self.slices:
            if pred_local == "xline":
                ref_section = reference[:, section // scale, :].T.astype(np.float64)
            else:
                ref_section = reference[section // scale, :, :].T.astype(np.float64)
            if scale == 2:
                ref_section = _upsample2(ref_section)[:n1, :n1]
            levels = np.percentile(ref_section, quantiles)
            horizons = _horizon_depths(_monotonic(ref_section), levels)
            keep = np.isfinite(horizons).mean(axis=1) >= 0.7
            self.ref_horizons[section] = horizons[keep]

    def evaluate(self, predicted_samples):
        """Return median deviations over all, deep, and individual depth bands."""
        bands = {index: [] for index in range(self.N_BANDS)}
        for section in self.slices:
            if section >= len(predicted_samples):
                continue
            prediction = _monotonic(
                np.asarray(predicted_samples[section]["pred"][0], dtype=np.float64)
            )
            horizons = self.ref_horizons[section]
            predicted_values = _sample_along(prediction, horizons)
            for index in range(horizons.shape[0]):
                valid = np.isfinite(predicted_values[index])
                if valid.sum() < prediction.shape[1] * 0.5:
                    continue
                level = np.median(predicted_values[index][valid])
                predicted_depth = _horizon_depths(prediction, np.array([level]))[0]
                residual = predicted_depth - horizons[index]
                residual = residual[np.isfinite(residual)]
                if residual.size < prediction.shape[1] * 0.5:
                    continue
                band = min(
                    int(np.nanmean(horizons[index]) // self.band), self.N_BANDS - 1
                )
                bands[band].append(np.std(residual - residual.mean()))

        median = lambda values: float(np.median(values)) if values else float("nan")
        band_medians = [median(bands[index]) for index in range(self.N_BANDS)]
        all_values = [value for index in range(self.N_BANDS) for value in bands[index]]
        deep_values = [
            value
            for index in (self.N_BANDS - 2, self.N_BANDS - 1)
            for value in bands[index]
        ]
        return {
            "all": median(all_values),
            "deep": median(deep_values),
            "bands": band_medians,
        }
