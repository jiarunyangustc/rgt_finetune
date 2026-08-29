# Synthetic example

`make_synthetic_case.py` creates a small time-domain seismic volume with
curved reflections and section-dependent structure. The generated directory
contains:

- `seismic.npy`: seismic amplitudes with shape `(24, 16, 64)`;
- `true_rgt.npy`: reference RGT used only to create the demonstration;
- `horizon_frame.dat`: two horizon surfaces encoded by constant values;
- `validation_horizon.npy`: an independently stored depth map;
- `segments.npy`: labeled local reflection patches;
- `pair_dip.npy`: adjacent-section depth shifts;
- `prediction_direct.dat`: a deliberately inconsistent section-wise result;
- `prediction_adapted.dat`: a more accurate, cross-section-consistent result;
- `sections/`: dictionaries following the training data contract.

Run:

```bash
python examples/make_synthetic_case.py --output examples/_synthetic_case
```

The script prints the output paths and basic shape checks. Evaluation of
`prediction_adapted.dat` should give a smaller validation-horizon MAE than
`prediction_direct.dat`:

```bash
for name in direct adapted; do
  python eval/eval_paper_protocol.py \
    --volume examples/_synthetic_case/prediction_${name}.dat \
    --shape 24,16,64 \
    --frame examples/_synthetic_case/horizon_frame.dat \
    --centers 0.25,0.72 \
    --input-idx 0 \
    --valid-maps examples/_synthetic_case/validation_horizon.npy
done
```

The example is deterministic for a fixed random seed.
