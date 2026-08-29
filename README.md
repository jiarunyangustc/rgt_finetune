# Multi-source constrained RGT fine-tuning

Official implementation accompanying the manuscript **“Fine-tuning a
pretrained network with multi-source information constraints for relative
geologic time estimation in field seismic data.”**

The method adapts a pretrained 2-D, section-wise relative geologic time (RGT)
network directly to a target 3-D seismic survey. It combines three sources of
survey information:

1. **Interpreted-horizon constraint:** RGT is encouraged to be constant along
   each sparse interpreted horizon, without prescribing an absolute RGT value.
2. **Horizon-segment constraint:** automatically tracked reflection patches
   provide denser local isochron constraints between interpreted horizons.
3. **Structural-orientation constraint:** local dips relate corresponding
   samples in adjacent sections and reduce discontinuities in the assembled
   3-D RGT volume.

The network processes one 2-D section at a time, but adjacent sections are
paired when the structural-orientation constraint is active. The paper's main
configuration trains rank-4 LoRA adapters and encoder convolution layers while
freezing the decoder.

## What is included

```text
train_finetune_only.py       training implementation and experiment runner
models_glp/                  MiT-b4 encoder, LoRA layers, and GLPN decoder
cigfaciesloss.py             horizon-segment losses
utils.py                     horizon/structure losses, samplers, training loop
configs/                     configurations used for the paper experiments
priors/                      segment tracking, local-dip estimation, data assembly
eval/eval_paper_protocol.py  volume-level horizon-MAE evaluation
examples/                    public synthetic example and tutorial
tests/                       lightweight regression tests
figures/                     scripts used to render manuscript figures
docs/                        data format and reproduction notes
```

The files in `configs/` preserve the experiment names used during method
development. Survey-specific names in file paths are internal identifiers and
do not disclose the field-data source.

## Installation

Python 3.9 and PyTorch 2.2 were used for the paper. A CUDA-enabled installation
is required for full-resolution training; the synthetic example and tests can
run on a CPU.

Using Conda:

```bash
conda env create -f environment.yml
conda activate rgt-finetune
```

Or install PyTorch for the desired CUDA version first, followed by the Python
dependencies:

```bash
pip install -r requirements.txt
```

`pyseistr` is needed only to estimate plane-wave-destruction dips. It is not
needed when a dip volume has already been prepared.

## Five-minute reproducibility check

The public example creates a small synthetic seismic volume, interpreted and
validation horizons, horizon-segment labels, local dips, and two illustrative
RGT predictions:

```bash
python examples/make_synthetic_case.py --output examples/_synthetic_case
python eval/eval_paper_protocol.py \
  --volume examples/_synthetic_case/prediction_adapted.dat \
  --shape 24,16,64 \
  --frame examples/_synthetic_case/horizon_frame.dat \
  --centers 0.25,0.72 \
  --input-idx 0 \
  --valid-maps examples/_synthetic_case/validation_horizon.npy
pytest -q
```

Expected behavior is documented in
[`examples/README.md`](examples/README.md). This example validates the public
data contract and evaluation workflow; it is intentionally small and is not a
substitute for the field-data experiments.

## Preparing a target survey

The trainer reads one NumPy dictionary per input section. At minimum, each
dictionary contains seismic amplitudes, the interpreted-horizon input, masks
for the interpreted horizons, segment labels, and a validity mask. The local
dip volume is stored once for the survey and referenced by the experiment
configuration.

See [`docs/data_format.md`](docs/data_format.md) and
`priors/make_dataset_example.py` for shapes, dtypes, coordinate conventions,
and a conversion example.

## Pretrained weights

Two weight files are required for field-data fine-tuning:

- `models_glp/weights/mit_b4.pth`: ImageNet-pretrained MiT-b4 encoder weights,
  released by the SegFormer authors. Download `mit_b4.pth` from the official
  [SegFormer repository](https://github.com/NVlabs/SegFormer#training) and
  place it under `models_glp/weights/`.
- `rgt_pretrained.pth`: the RGT-pretrained checkpoint (702 MB;
  MD5 `8f22bb5286718461debbf33566cf51c9`), available from this
  repository's [Releases page](https://github.com/jiarunyangustc/rgt_finetune/releases).
  Point `pretrained_checkpoint` in the experiment configuration to it.

The training code raises a clear error when the RGT-pretrained checkpoint is
absent instead of silently starting from random weights.

## Running a paper configuration

Each Python file in `configs/` defines one experiment and calls
`run_experiment`:

```bash
python configs/run_512_segqc3_pairdip_lr1e4_sr3_ep150.py
```

Before running it, update these configuration entries for the local system:

- `train_sample_path`;
- `pair_dip_path`;
- `pretrained_checkpoint`;
- `output_dir` and `checkpoint_root` if the defaults are unsuitable.

The main experiments use 150 epochs, an initial learning rate of `1e-4`, a
batch of 40 sections arranged as 20 adjacent pairs, and loss weights
`(1.0, 0.3, 0.1)` for interpreted horizons, horizon segments, and structural
orientation, respectively.

## Evaluation protocol

For every evaluation horizon, one RGT value is assigned over the complete 3-D
volume. The horizon is extracted from each trace by linear interpolation of
the inverse RGT-depth relation. Mean absolute vertical errors are first
computed per horizon; reported input- and validation-horizon errors are
equal-weight averages over the horizons in each group.

Run `python eval/eval_paper_protocol.py --help` for all options.

## Field data and reproducibility

The two field seismic surveys and their interpretations are subject to data
use restrictions and cannot be redistributed. Consequently, exact field-data
figure reproduction requires authorized access to those surveys. To support
independent assessment, this repository provides:

- the complete model, loss, fine-tuning, prior-extraction, and evaluation code;
- all paper experiment configurations;
- a synthetic test case following the same array and coordinate conventions;
- tests for the principal loss behavior and evaluation workflow.

See [`docs/reproduction.md`](docs/reproduction.md) for the mapping between
manuscript results and configuration files.

## License and citation

The authors' original code is released under the MIT License. Adapted model
components retain upstream non-commercial research restrictions. See
[`LICENSE`](LICENSE), [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md), and
source-file headers before redistribution or reuse.

Citation metadata are provided in [`CITATION.cff`](CITATION.cff).
