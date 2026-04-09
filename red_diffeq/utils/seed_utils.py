"""Seed utilities for reproducible experiments."""

import os
import random
import numpy as np
import torch


def set_seed(seed: int, verbose: bool = True, allow_tf32: bool = False):
    """
    Set random seeds and deterministic flags.

    Args:
        seed: Random seed value
        verbose: Print status messages
        allow_tf32: If True, allow TF32 math on supported GPUs. If False,
            disable TF32 for tighter cross-device numerical consistency.
    """
    # Python random module
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA (all GPUs)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # cuDNN deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # For PyTorch >= 1.8: Enable deterministic algorithms
    # This makes CUDA operations deterministic but may be slower
    try:
        torch.use_deterministic_algorithms(True)
    except AttributeError:
        # PyTorch < 1.8 doesn't have this function
        pass

    # Set environment variables for full CUDA determinism
    # CUBLAS_WORKSPACE_CONFIG is required for some operations
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    # TF32 control on supported GPUs.
    if hasattr(torch.backends.cuda, 'matmul'):
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = allow_tf32

    if verbose:
        print(f"   - Random seed set to: {seed}")
        print(f"   - Python random: {seed}")
        print(f"   - NumPy: {seed}")
        print(f"   - PyTorch CPU: {seed}")
        if torch.cuda.is_available():
            print(f"   - PyTorch CUDA (all devices): {seed}")
        print(f"   - cuDNN deterministic: True")
        print(f"   - cuDNN benchmark: False")
        try:
            if torch.are_deterministic_algorithms_enabled():
                print(f"   - Deterministic algorithms: Enabled")
        except (AttributeError, RuntimeError):
            pass
        print(f"   - CUBLAS_WORKSPACE_CONFIG: :4096:8")
        if hasattr(torch.backends.cuda, 'matmul'):
            if allow_tf32:
                print(f"   - TF32 enabled")
            else:
                print(f"   - TF32 disabled")


def worker_init_fn(worker_id: int, base_seed: int = 0):
    """
    Worker initialization function for DataLoader to ensure reproducibility.

    Args:
        worker_id: Worker ID assigned by DataLoader
        base_seed: Base seed to use (usually the global seed)

    Usage:
        DataLoader(..., worker_init_fn=lambda id: worker_init_fn(id, seed))
    """
    worker_seed = base_seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def get_rng_state():
    """
    Get current RNG state for all random number generators.

    Returns:
        Dictionary containing all RNG states
    """
    state = {
        'python_random': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()

    return state


def set_rng_state(state: dict):
    """
    Restore RNG state for all random number generators.

    Args:
        state: Dictionary containing RNG states (from get_rng_state())
    """
    random.setstate(state['python_random'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])

    if torch.cuda.is_available() and 'cuda' in state:
        torch.cuda.set_rng_state_all(state['cuda'])


class SeedContext:
    """
    Context manager for temporary seed changes.

    Usage:
        with SeedContext(42):
            # All random operations use seed 42
            x = torch.randn(10)
        # Original RNG state restored
    """

    def __init__(self, seed: int):
        self.seed = seed
        self.saved_state = None

    def __enter__(self):
        self.saved_state = get_rng_state()
        set_seed(self.seed, verbose=False)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        set_rng_state(self.saved_state)
        return False
