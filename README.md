# Multi-source constrained RGT fine-tuning

Code for the paper *"Fine-tuning a pretrained network with multi-source
information constraints for relative geologic time estimation in field
seismic data"* (submitted to Computers & Geosciences).

A pretrained section-wise RGT network (transformer encoder + convolutional
decoder) is fine-tuned directly on the target survey with three
complementary constraints derived from the survey itself:

1. **Horizon constraint** — a few interpreted horizons (uniform RGT along
   each surface; no target value prescribed);
2. **Segment constraint** — reflection segments tracked automatically in
   3-D (uniform RGT along each labeled segment, smooth-L1, detached center);
3. **Structure constraint** — dip-compensated consistency between
   neighboring section predictions (smooth-L1 on
   tau_i(z, x) - tau_{i+1}(z + delta, x), dips from plane-wave destruction).

Fine-tuning updates the encoder only (rank-4 LoRA on all linear layers,
full update of all encoder convolutions); the decoder is frozen.

## Repository layout

```
train_finetune_only.py    entry point: builds dataset, freezes weights, trains
utils.py                  training loop, losses assembly, pair batch sampler
cigfaciesloss.py          SegmentLoss (smooth-L1, detached center)
models_glp/               network (MiT-b4 encoder + GLPN decoder, LoRA hooks)
draw.py, tools.py         plotting / helper utilities used by the trainer
valid_truth.py            loaders for independent validation picks

priors/
  generate_segments_3d.py            3-D waveform-correlation segment tracker
  generate_zxdata_512_segments_*.py  survey-2 variant (fault barrier)
  make_pair_dip_lunnan.py            cross-section dips (PWD) + sign check, survey 1
  make_pair_dip_512.py               cross-section dips for the upsampled grid
  diag_pwd_dip_qc.py                 native-grid PWD dip estimation + QC
  make_dataset_example.py            per-section training-sample assembly example

configs/
  run_lunnan_h13_segamp_pairdip_sr3_ep150.py   survey 1, full scheme (paper)
  run_256_lossonly_pairdip_sr3_ep150.py        survey 2, 256^3, full scheme
  run_512_segqc3_pairdip_lr1e4_sr3_ep150.py    survey 2, 512^3, full scheme
  run_*_olhr_* / run_*_segonly_*               chain members (horizon / +segments)
  ablation_*                                   which-weights / batch-size / lr ablations

eval/eval_paper_protocol.py   horizon MAE under the paper protocol
figures/                      figure-rendering scripts used in the paper
```

## Installation

```
pip install torch numpy scipy scikit-image matplotlib tqdm pillow loralib
pip install mmcv            # only for loading the ImageNet-pretrained backbone
pip install pyseistr        # plane-wave destruction (dip estimation)
```

Tested with Python 3.9 and PyTorch 2.2 (CUDA 12.1).

## Data and pretrained weights

Field seismic volumes are not redistributed here. The code expects, per
survey (float32, layout `(n1, n2, depth)`):

- the seismic volume;
- a horizon "frame" volume in which each interpreted horizon is marked by a
  constant value on its samples;
- a labeled segment volume from `priors/generate_segments_3d.py`;
- a cross-section dip volume from `priors/make_pair_dip_*.py`.

`priors/make_dataset_example.py` shows how these are assembled into the
per-section training samples read by the trainer.

The pretrained network weights (ImageNet MiT-b4 backbone and the
RGT-pretrained checkpoint) are archived separately (see the paper's data
availability statement); place them under `models_glp/weights/` and
`checkpoints/` respectively.

## Running

Each config is a self-contained experiment description:

```
python configs/run_lunnan_h13_segamp_pairdip_sr3_ep150.py
```

All main experiments share one recipe: 150 epochs, batch of 40 sections
(paired-adjacent sampler when the structure constraint is on), cosine
schedule, lr 1e-4, loss weights (1, 0.3, 0.1), identical trainable layers.

Evaluation (protocol of the paper — one global RGT value per horizon,
equal-weight groups):

```
python eval/eval_paper_protocol.py --volume pred.dat --shape 416,256,256 \
    --frame frame.dat --centers 0.2638,0.4061,0.7778,0.8445 \
    --input-idx 0,2 --valid-idx 1,3
```

## Notes

- `utils.py` is the research-grade trainer used for the paper, including
  code paths for ablations that are switched off by default; the configs in
  `configs/` document exactly which switches each experiment uses.
- The `epoch_hz_px` entry in the saved training logs is a passive
  depth-equivalent horizon-misfit monitor (immune to gradient shrinkage);
  it is diagnostic only and never used for model selection.

## License

MIT (see LICENSE).
