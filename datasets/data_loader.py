"""PyTorch Dataset & DataLoader utilities for spatiotemporal wildfire forecasting."""

from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config import CONFIG
from utils.logger import get_logger
from utils.validation import validate_input_tensor, validate_target_tensor
from datasets.normalization import SpatioTemporalNormalizer

logger = get_logger("data_loader")


class WildfireDataset(Dataset):
    """PyTorch Dataset for multi-modal spatiotemporal wildfire sequence-to-sequence forecasting."""

    def __init__(
        self,
        inputs: np.ndarray | torch.Tensor,
        targets: np.ndarray | torch.Tensor,
        normalizer: Optional[SpatioTemporalNormalizer] = None,
        augment: bool = False
    ):
        """
        Args:
            inputs: (N, 5, 6, 128, 128)
            targets: (N, 3, 1, 128, 128)
            normalizer: Optional feature normalizer instance.
            augment: If True, applies random geometric spatial symmetries (rot90, flip).
        """
        if isinstance(inputs, np.ndarray):
            inputs = torch.from_numpy(inputs).float()
        if isinstance(targets, np.ndarray):
            targets = torch.from_numpy(targets).float()

        self.inputs = inputs
        self.targets = targets
        self.normalizer = normalizer
        self.augment = augment

        # Perform strict dimensional validation
        validate_input_tensor(self.inputs)
        validate_target_tensor(self.targets)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.inputs[idx]  # (5, 6, 128, 128)
        y = self.targets[idx] # (3, 1, 128, 128)

        if self.normalizer is not None:
            x = self.normalizer.normalize_tensor(x)

        if self.augment:
            # Spatial data augmentation consistent across time & channels
            if torch.rand(1).item() > 0.5:
                x = torch.flip(x, dims=[-1])
                y = torch.flip(y, dims=[-1])
            if torch.rand(1).item() > 0.5:
                x = torch.flip(x, dims=[-2])
                y = torch.flip(y, dims=[-2])
            k = int(torch.randint(0, 4, (1,)).item())
            if k > 0:
                x = torch.rot90(x, k, dims=[-2, -1])
                y = torch.rot90(y, k, dims=[-2, -1])

        return x, y


def load_dataset_from_npz(npz_path: Path | str) -> Tuple[np.ndarray, np.ndarray]:
    """Loads inputs and targets from compressed npz file."""
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"Dataset archive not found: {npz_path}")

    data = np.load(npz_path)
    inputs = data["inputs"].astype(np.float32)
    targets = data["targets"].astype(np.float32)
    return inputs, targets


def create_dataloaders(
    dataset_path: Optional[Path | str] = None,
    inputs: Optional[np.ndarray] = None,
    targets: Optional[np.ndarray] = None,
    batch_size: int = CONFIG.data.batch_size,
    train_ratio: float = CONFIG.data.train_ratio,
    val_ratio: float = CONFIG.data.val_ratio,
    test_ratio: float = CONFIG.data.test_ratio,
    seed: int = CONFIG.data.seed,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """Creates non-leaking Train, Validation, and Test DataLoaders.
    
    Returns:
        train_loader, val_loader, test_loader, metadata_dict
    """
    if inputs is None or targets is None:
        if dataset_path is None:
            dataset_path = CONFIG.dataset_path
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            from datasets.real_dataset_builder import RealWildfireDatasetBuilder
            builder = RealWildfireDatasetBuilder(num_sequences=120)
            builder.build_and_save(dataset_path)
        inputs, targets = load_dataset_from_npz(dataset_path)

    total_samples = len(inputs)
    rng = np.random.default_rng(seed)
    indices = np.arange(total_samples)
    rng.shuffle(indices)

    n_train = int(total_samples * train_ratio)
    n_val = int(total_samples * val_ratio)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    normalizer = SpatioTemporalNormalizer()

    train_ds = WildfireDataset(inputs[train_idx], targets[train_idx], normalizer=normalizer, augment=True)
    val_ds = WildfireDataset(inputs[val_idx], targets[val_idx], normalizer=normalizer, augment=False)
    test_ds = WildfireDataset(inputs[test_idx], targets[test_idx], normalizer=normalizer, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )

    metadata = {
        "total_samples": total_samples,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "input_shape": list(inputs.shape),
        "target_shape": list(targets.shape),
    }

    logger.info(
        f"DataLoaders created - Train: {metadata['n_train']}, Val: {metadata['n_val']}, Test: {metadata['n_test']} samples."
    )
    return train_loader, val_loader, test_loader, metadata

