# Third-party notices

This repository contains or adapts components from the projects below. The
root MIT license applies only to the authors' original contributions and does
not replace these upstream terms.

## SegFormer / Mix Transformer

Files `models_glp/mit.py` and `models_glp/mit_peg.py` are derived from the
official SegFormer implementation:

- Source: <https://github.com/NVlabs/SegFormer>
- Copyright: NVIDIA Corporation
- License: NVIDIA Source Code License for SegFormer
- Local license copy: [`licenses/SEGFORMER_LICENSE`](licenses/SEGFORMER_LICENSE)

The upstream license limits use to non-commercial research or evaluation and
requires redistribution under that license with notices retained.

## GLPDepth

The decoder design and parts of `models_glp/model.py` are adapted from the
official GLPDepth research implementation:

- Source: <https://github.com/vinvino02/GLPDepth>
- Authors: Doyeon Kim and collaborators
- Upstream terms: non-commercial research and evaluation only, as stated in
  the upstream README

## LoRA

Low-rank adapter layers use the `loralib` Python package:

- Source: <https://github.com/microsoft/LoRA>
- License: MIT

## pyseistr

Plane-wave-destruction dip estimation uses `pyseistr` as an external
dependency and does not vendor its source:

- Source: <https://github.com/aaspip/pyseistr>
- License: MIT

Users are responsible for ensuring that their use complies with all applicable
upstream terms. For a fully permissive software release, the SegFormer and
GLPDepth-derived components must be replaced by license-compatible
implementations and checkpoint compatibility must be revalidated.
