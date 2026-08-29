# Reproducing manuscript experiments

## Configuration mapping

| Manuscript experiment | Configuration |
|---|---|
| Survey 1, interpreted horizons | `configs/run_lunnan_h13_olhr_ep150.py` |
| Survey 1, horizons + segments | `configs/run_lunnan_h13_segamp_only_ep150.py` |
| Survey 1, complete framework | `configs/run_lunnan_h13_segamp_pairdip_sr3_ep150.py` |
| Survey 2, 256-cube complete framework | `configs/run_256_lossonly_pairdip_sr3_ep150.py` |
| Survey 2, 512-cube interpreted horizons | `configs/run_512_olhr_a000_ep150_lr1e4.py` |
| Survey 2, 512-cube horizons + segments | `configs/run_512_segqc3_only_a03_ep150_lr1e4.py` |
| Survey 2, 512-cube complete framework | `configs/run_512_segqc3_pairdip_lr1e4_sr3_ep150.py` |
| Fine-tuning-range ablations | `configs/ablation_*loraonly.py`, `configs/ablation_*fullft.py`, and `configs/ablation_*sr12*.py` |
| Mini-batch ablations | `configs/ablation_512_bs*.py` |

## Reproducibility levels

1. **Public smoke reproduction:** run the synthetic example and `pytest -q`.
   This checks the array contract, horizon-segment loss, horizon loss, and
   evaluation protocol without restricted data or a GPU.
2. **Model reproduction:** download the released MiT and RGT checkpoints,
   prepare data following `docs/data_format.md`, and run a configuration.
3. **Field-result reproduction:** requires authorized copies of the two field
   surveys and interpreted horizons. Their redistribution is prohibited by
   the data owners.

## Release checklist

Before creating the archival GitHub/Zenodo release, replace the placeholders
below with permanent links and SHA-256 checksums:

| Artifact | URL | SHA-256 |
|---|---|---|
| RGT-pretrained checkpoint | `TO_BE_ADDED` | `TO_BE_ADDED` |
| MiT-b4 checkpoint | `TO_BE_ADDED` | `TO_BE_ADDED` |
| Synthetic example archive (optional) | generated locally | not applicable |

Record the Git commit and software environment with every reproduced result.
