# RED-DiffEq Configuration Files

This directory contains YAML files used by `scripts/run_inversion.py`.

## Run With A Config

```bash
python scripts/run_inversion.py --config configs/default.yaml
```

## Available Configs

- `configs/default.yaml`: Default baseline settings.
- `configs/openfwi/red-diffeq.yaml`: OpenFWI inversion setup.
- `configs/marmousi/red-diffeq.yaml`: Marmousi inversion setup.
- `configs/marmousi/tv.yaml`: Marmousi with TV regularization.
- `configs/marmousi/tikhonov.yaml`: Marmousi with L2 (Tikhonov) regularization.
- `configs/overthrust/red-diffeq.yaml`: Overthrust inversion setup.
- `configs/overthrust/tv.yaml`: Overthrust with TV regularization.
- `configs/overthrust/tikhonov.yaml`: Overthrust with L2 (Tikhonov) regularization.

## Notes

- Current maintained diffusion checkpoint path is `pretrained_models/model-4.pt`.
- For reproducibility, set `experiment.random_seed` in the config.
- Paths in `data.seismic_data_dir` and `data.velocity_data_dir` are relative to repo root.
