import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse

import numpy as np
import torch
from tqdm import tqdm
from accelerate import Accelerator
from torch.optim import Adam

from red_diffeq import (
    load_config,
    save_config,
    GaussianDiffusion,
    Unet,
    FWIForward,
    InversionEngine,
    SSIM,
    prepare_initial_model,
    s_normalize_none,
    v_denormalize,
)
import ml_collections


def setup_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    return device


def load_diffusion_model(config: ml_collections.ConfigDict, device: torch.device) -> GaussianDiffusion:
    """Load diffusion model with Accelerator and mixed precision."""
    model = Unet(
        dim=config.model.dim,
        dim_mults=config.model.dim_mults,
        flash_attn=config.model.flash_attn,
        channels=config.model.channels,
    )

    diffusion = GaussianDiffusion(
        model,
        image_size=config.diffusion.image_size,
        timesteps=config.diffusion.timesteps,
        sampling_timesteps=config.diffusion.sampling_timesteps,
        objective=config.diffusion.objective,
    ).to(device)

    accelerator = Accelerator(
        split_batches=True,
        mixed_precision='fp16'
    )
    
    opt = Adam(diffusion.parameters(), lr=20, betas=(0.9, 0.99))
    diffusion, opt = accelerator.prepare(diffusion, opt)
    diffusion = accelerator.unwrap_model(diffusion)

    model_path = Path(config.diffusion.model_path)
    if model_path.exists():
        checkpoint = torch.load(model_path, map_location=device)
        diffusion.load_state_dict(checkpoint["model"])
        print(f"Loaded pretrained model from: {model_path}")
    else:
        print(f"WARNING: Pretrained model not found at {model_path}")
        print("Continuing with randomly initialized model...")

    diffusion.eval()
    return diffusion


def initialize_forward_operator(config: ml_collections.ConfigDict, device: torch.device) -> FWIForward:
    ctx = config.pde.to_dict()

    fwi_forward = FWIForward(
        ctx,
        device,
        normalize=True,
        v_denorm_func=v_denormalize,
        s_norm_func=s_normalize_none,
    )

    return fwi_forward


def get_data_files(config: ml_collections.ConfigDict) -> list:
    seismic_dir = Path(config.data.seismic_data_dir)

    if not seismic_dir.exists():
        raise FileNotFoundError(f"Seismic data directory not found: {seismic_dir}")

    pattern = config.data.data_pattern
    family_files = sorted(seismic_dir.glob(pattern))

    if not family_files:
        raise ValueError(f"No data files found matching {pattern} in {seismic_dir}")

    all_families = [f.name for f in family_files]

    openfwi_families = getattr(config.data, 'openfwi_families', None)
    if openfwi_families is None or openfwi_families == []:
        return all_families

    if isinstance(openfwi_families, str):
        openfwi_families = [openfwi_families]

    filtered = [f if f.endswith('.npy') else f"{f}.npy"
                for f in openfwi_families if f is not None]

    if not filtered:
        return all_families

    result = [f for f in all_families if f in filtered]

    if not result:
        raise ValueError(
            f"No matching families found. Requested: {filtered}, "
            f"Available: {all_families}"
        )

    return result


def process_batch(
    batch_start: int,
    batch_end: int,
    seis_mmap: np.ndarray,
    vel_mmap: np.ndarray,
    config: ml_collections.ConfigDict,
    inversion_engine: InversionEngine,
    fwi_forward: FWIForward,
    device: torch.device,
) -> tuple:
    current_batch_size = batch_end - batch_start

    seis_batch = torch.from_numpy(seis_mmap[batch_start:batch_end].copy()).float().to(device)
    vel_batch = torch.from_numpy(vel_mmap[batch_start:batch_end].copy()).float()

    initial_models = []
    for i in range(current_batch_size):
        vel_slice = vel_batch[i : i + 1]
        initial_model = prepare_initial_model(
            vel_slice,
            config.optimization.initial_type,
            sigma=config.optimization.sigma,
        )
        # Pad by one cell on each side for the diffusion model input shape.
        initial_model = torch.nn.functional.pad(initial_model, (1, 1, 1, 1), "constant", 0)
        initial_models.append(initial_model)

    initial_model_batch = torch.cat(initial_models, dim=0)

    mu_batch, final_results_per_model = inversion_engine.optimize(
        initial_model_batch,
        vel_batch,
        seis_batch,
        fwi_forward,
        ts=config.optimization.ts,
        lr=config.optimization.lr,
        reg_lambda=config.optimization.reg_lambda,
        noise_std=config.optimization.noise_std,
        noise_type=config.optimization.noise_type,
        missing_number=config.optimization.missing_number,
        regularization=config.optimization.regularization
        if config.optimization.regularization and config.optimization.regularization != "none"
        else None,
    )

    return mu_batch, final_results_per_model, initial_model_batch, vel_batch


def save_batch_results(
    batch_start: int,
    batch_end: int,
    mu_batch: torch.Tensor,
    results_per_model: list,
    initial_model_batch: torch.Tensor,
    vel_batch: torch.Tensor,
    output_dir: Path,
) -> None:

    mu_batch_np = mu_batch.detach().cpu().numpy()
    vel_batch_np = vel_batch.cpu().numpy()
    initial_model_batch_np = initial_model_batch[:, :, 1:-1, 1:-1].detach().cpu().numpy()

    for i, model_idx in enumerate(range(batch_start, batch_end)):
        mu_result_2d = mu_batch_np[i, 0, :, :]
        initial_velocity_2d = initial_model_batch_np[i, 0, :, :]
        ground_truth_2d = vel_batch_np[i, 0, :, :]
        model_metrics = results_per_model[i]

        npz_data = {
            "result": mu_result_2d,
            "initial_velocity": initial_velocity_2d,
            "ground_truth": ground_truth_2d,
            "total_losses": np.array(model_metrics["total_losses"]),
            "obs_losses": np.array(model_metrics["obs_losses"]),
            "reg_losses": np.array(model_metrics["reg_losses"]),
            "ssim": np.array(model_metrics["ssim"]),
            "mae": np.array(model_metrics["mae"]),
            "rmse": np.array(model_metrics["rmse"]),
        }

        npz_path = output_dir / f"{model_idx}_results.npz"
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(npz_path.resolve()), **npz_data)


def run_experiment(config: ml_collections.ConfigDict) -> None:
    base_seed = config.experiment.random_seed
    if base_seed is not None:
        from red_diffeq.utils.seed_utils import set_seed
        set_seed(base_seed, verbose=True)
    else:
        print("No random seed set - experiment will be non-deterministic")

    print("\n" + "=" * 70)
    print("Configuration:")
    print("=" * 70)
    for key, value in sorted(config.items()):
        if not isinstance(value, ml_collections.ConfigDict):
            print(f"  {key}: {value}")
    print("=" * 70 + "\n")

    device = setup_device()

    print("Initializing models...")
    diffusion = load_diffusion_model(config, device)
    fwi_forward = initialize_forward_operator(config, device)

    ssim_loss = SSIM(window_size=11, size_average=True)
    inversion_engine = InversionEngine(
        diffusion,
        ssim_loss,
        config.optimization.regularization if config.optimization.regularization else None,
        use_time_weight=getattr(config.optimization, 'use_time_weight', False),
        sigma_x0=getattr(config.optimization, 'sigma_x0', 0.0001),
        fixed_timestep=getattr(config.optimization, 'fixed_timestep', None),
    )

    seismic_dir = Path(config.data.seismic_data_dir).resolve()
    dataset_name = seismic_dir.parts[-2] if len(seismic_dir.parts) >= 2 else None

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if dataset_name:
        results_dir = Path(config.experiment.results_dir) / dataset_name / config.experiment.name / timestamp
    else:
        results_dir = Path(config.experiment.results_dir) / config.experiment.name / timestamp
    print(f"Results will be saved to: {results_dir}")
    results_dir.mkdir(parents=True, exist_ok=True)

    config_save_path = results_dir / "config.yaml"
    save_config(config, config_save_path)
    print(f"Configuration saved to: {config_save_path}")

    print("Loading data files...")
    family_files = get_data_files(config)
    print(f"Found {len(family_files)} data families to process")

    for family_name in family_files:
        print(f"\n{'='*70}")
        print(f"Processing: {family_name}")
        print(f"{'='*70}")

        family_results_dir = results_dir / Path(family_name).stem
        family_results_dir.mkdir(exist_ok=True)

        seismic_path = Path(config.data.seismic_data_dir) / family_name
        velocity_path = Path(config.data.velocity_data_dir) / family_name

        seis_mmap = np.load(seismic_path, mmap_mode="r")
        vel_mmap = np.load(velocity_path, mmap_mode="r")
        num_models = seis_mmap.shape[0]

        sample_index = getattr(config.data, 'sample_index', None)
        if sample_index is not None:
            if sample_index < 0 or sample_index >= num_models:
                print(f"Warning: sample_index {sample_index} is out of range [0, {num_models-1}]. Skipping {family_name}.")
                continue
            print(f"Processing only sample {sample_index} (out of {num_models} samples)")
            batch_start = sample_index
            batch_end = sample_index + 1
            num_batches = 1
        else:
            print(f"Number of models: {num_models}")
            print(f"Batch size: {config.data.batch_size}")
            num_batches = (num_models + config.data.batch_size - 1) // config.data.batch_size

        for batch_idx in tqdm(range(num_batches), desc="Batches"):
            if sample_index is None:
                batch_start = batch_idx * config.data.batch_size
                batch_end = min(batch_start + config.data.batch_size, num_models)

            mu_batch, results, initial_batch, vel_batch = process_batch(
                batch_start,
                batch_end,
                seis_mmap,
                vel_mmap,
                config,
                inversion_engine,
                fwi_forward,
                device,
            )

            save_batch_results(
                batch_start,
                batch_end,
                mu_batch,
                results,
                initial_batch,
                vel_batch,
                family_results_dir,
            )

    print(f"\n{'='*70}")
    print(f"Experiment complete! Results saved to: {results_dir}")
    print(f"{'='*70}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Full Waveform Inversion with RED-DiffEq",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML configuration file",
    )

    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--ts", type=int, help="Number of optimization steps")
    parser.add_argument(
        "--regularization",
        choices=["diffusion", "tv", "l2", "none"],
        help="Regularization type",
    )
    parser.add_argument("--reg_lambda", type=float, help="Regularization weight")
    parser.add_argument(
        "--noise_type",
        choices=["gaussian", "laplace"],
        help="Noise type: 'gaussian' (default) or 'laplace'",
    )
    parser.add_argument("--noise_std", type=float, help="Noise standard deviation (Gaussian) or scale (Laplace)")
    parser.add_argument("--sigma", type=float, help="Initial model smoothing sigma")
    parser.add_argument("--sigma_x0", type=float, help="Pre-noise added to mu before diffusion forward process (for noise robustness, default: 0.0001)")
    parser.add_argument("--missing_number", type=int, help="Number of missing traces")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--experiment_name", type=str, help="Experiment name")
    parser.add_argument(
        "--results_dir",
        type=Path,
        help="Base directory where results will be written (overrides config.experiment.results_dir)",
    )
    parser.add_argument("--random_seed", type=int, help="Random seed")
    parser.add_argument("--openfwi_families", type=str, nargs="+", help="OpenFWI families to process (e.g., CF CV or CF.npy CV.npy). Default: process all families")
    parser.add_argument("--sample_index", type=int, default=None, help="Process only a specific sample index (0-indexed). Default: process all samples")
    args = parser.parse_args()

    if args.config:
        print(f"Loading configuration from: {args.config}")
        config = load_config(args.config)
    else:
        print("No config file specified. Using default configuration.")
        from red_diffeq import get_config
        config = get_config()

    if args.lr is not None:
        config.optimization.lr = args.lr
    if args.ts is not None:
        config.optimization.ts = args.ts
    if args.regularization is not None:
        config.optimization.regularization = args.regularization
    if args.reg_lambda is not None:
        config.optimization.reg_lambda = args.reg_lambda
    if args.results_dir is not None:
        config.experiment.results_dir = str(args.results_dir)
    if args.noise_type is not None:
        config.optimization.noise_type = args.noise_type
    if args.noise_std is not None:
        config.optimization.noise_std = args.noise_std
    if args.sigma is not None:
        config.optimization.sigma = args.sigma
    if args.sigma_x0 is not None:
        config.optimization.sigma_x0 = args.sigma_x0
    if args.missing_number is not None:
        config.optimization.missing_number = args.missing_number
    if args.batch_size is not None:
        config.data.batch_size = args.batch_size
    if args.experiment_name is not None:
        config.experiment.name = args.experiment_name
    if args.random_seed is not None:
        config.experiment.random_seed = args.random_seed
    if args.openfwi_families is not None:
        config.data.openfwi_families = args.openfwi_families
    
    if args.sample_index is not None:
        config.data.sample_index = args.sample_index
    run_experiment(config)


if __name__ == "__main__":
    main()
