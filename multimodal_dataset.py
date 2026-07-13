"""
MultiModal Dataset for GPS, LiDAR, and RGB Image data.

This module provides a PyTorch Dataset class for loading and processing
multimodal sensor data with sliding window sampling strategy.

"""

import os
from typing import Optional, Dict, List, Tuple, Callable

import numpy as np
import pandas as pd
import scipy.io
import torch
from PIL import Image
from torch.utils.data import Dataset

# Try to import torchvision, fall back to manual transforms if not available
try:
    from torchvision import transforms
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False


def load_gps(gps_path: str) -> np.ndarray:
    """
    Load GPS coordinates from text file.

    File format: Two lines
        Line 1: latitude (float)
        Line 2: longitude (float)

    Args:
        gps_path: Path to the GPS text file

    Returns:
        numpy array of shape (2,) containing [latitude, longitude]
    """
    with open(gps_path, 'r') as f:
        lines = f.readlines()

    latitude = float(lines[0].strip())
    longitude = float(lines[1].strip())

    return np.array([latitude, longitude], dtype=np.float32)


def load_lidar(lidar_path: str) -> np.ndarray:
    """
    Load LiDAR data from .mat file.

    The .mat file contains a 'data' key with shape (216, 2).
    We extract the first column which contains the range/intensity values.

    Args:
        lidar_path: Path to the .mat file

    Returns:
        numpy array of shape (216,) containing LiDAR measurements
    """
    mat_data = scipy.io.loadmat(lidar_path)
    # Extract first column: (216, 2) -> (216,)
    lidar_data = mat_data['data'][:, 0].astype(np.float32)
    return lidar_data


def load_image(image_path: str, transform: Optional[Callable] = None) -> torch.Tensor:
    """
    Load RGB image from file.

    Args:
        image_path: Path to the image file
        transform: Optional transform to apply

    Returns:
        Tensor of shape (3, 540, 960)
    """
    img = Image.open(image_path).convert('RGB')

    if transform is not None:
        img = transform(img)
    elif HAS_TORCHVISION:
        # Use torchvision transforms if available
        img = transforms.ToTensor()(img)  # (3, 540, 960)
    else:
        # Manual conversion: PIL Image -> numpy -> torch tensor
        # Convert to numpy array: (H, W, C) with values in [0, 255]
        img_np = np.array(img, dtype=np.float32) / 255.0
        # Transpose to (C, H, W) format
        img = torch.from_numpy(img_np.transpose(2, 0, 1))

    return img


class MultiModalDataset(Dataset):
    """
    PyTorch Dataset for multimodal sensor data (GPS, LiDAR, RGB Image).

    Uses sliding window sampling within sequences, with edge dropping
    to avoid boundary effects.

    Args:
        csv_path: Path to the CSV index file
        root_dir: Root directory containing the data
        split: 'train' or 'test'
        train_ratio: Ratio of sequences for training (default: 0.8)
        window_size: Number of consecutive samples in each window (default: 4)
        edge_drop: Number of samples to drop from each end of sequence (default: 3)
        transform: Optional transform for images
        normalize_gps: Whether to normalize GPS coordinates (default: True)
        gps_stats: Pre-computed GPS stats {'mean': ndarray, 'std': ndarray} for test set
        random_seed: Random seed for reproducible train/test split (default: 42)
        verbose: Print dataset statistics if True

    Example:
        >>> train_dataset = MultiModalDataset(
        ...     csv_path='scenario9.csv',
        ...     root_dir='scenario9_dev',
        ...     split='train',
        ...     train_ratio=0.8
        ... )
        >>> # Use train stats for test set
        >>> test_dataset = MultiModalDataset(
        ...     csv_path='scenario9.csv',
        ...     root_dir='scenario9_dev',
        ...     split='test',
        ...     train_ratio=0.8,
        ...     gps_stats=train_dataset.get_gps_stats()
        ... )
    """

    def __init__(
        self,
        csv_path: str,
        root_dir: str,
        split: str = 'train',
        train_ratio: float = 0.8,
        window_size: int = 4,
        edge_drop: int = 3,
        transform: Optional[Callable] = None,
        normalize_gps: bool = True,
        gps_stats: Optional[Dict[str, np.ndarray]] = None,
        random_seed: int = 42,
        verbose: bool = True
    ):
        assert split in ['train', 'test'], f"split must be 'train' or 'test', got '{split}'"
        assert 0 < train_ratio < 1, f"train_ratio must be between 0 and 1, got {train_ratio}"

        self.csv_path = csv_path
        self.root_dir = root_dir
        self.split = split
        self.train_ratio = train_ratio
        self.window_size = window_size
        self.edge_drop = edge_drop
        self.transform = transform
        self.normalize_gps = normalize_gps
        self.random_seed = random_seed
        self.verbose = verbose

        # Load CSV file
        self.df = pd.read_csv(csv_path)

        # Split by percentage based on seq_index
        self._split_by_ratio()

        # Build window indices
        self.windows = self._build_windows()

        # Setup GPS normalization
        self.gps_mean = None
        self.gps_std = None
        if normalize_gps:
            if gps_stats is not None:
                # Use provided stats (for test set, use train stats)
                self.gps_mean = gps_stats['mean']
                self.gps_std = gps_stats['std']
            else:
                # Compute stats from current dataset (should be train set)
                self._compute_gps_stats()

        if verbose:
            self._print_statistics()

    def _split_by_ratio(self):
        """Split sequences by train_ratio percentage."""
        # Get unique seq_indices
        all_seq_indices = sorted(self.df['seq_index'].unique())
        n_sequences = len(all_seq_indices)

        # Shuffle with fixed seed for reproducibility
        np.random.seed(self.random_seed)
        shuffled_indices = np.random.permutation(all_seq_indices)

        # Split
        n_train = int(n_sequences * self.train_ratio)
        train_seq_indices = set(shuffled_indices[:n_train])
        test_seq_indices = set(shuffled_indices[n_train:])

        # Store for reference
        self.train_seq_indices = sorted(train_seq_indices)
        self.test_seq_indices = sorted(test_seq_indices)

        # Filter DataFrame
        if self.split == 'train':
            self.df = self.df[self.df['seq_index'].isin(train_seq_indices)].reset_index(drop=True)
        else:
            self.df = self.df[self.df['seq_index'].isin(test_seq_indices)].reset_index(drop=True)

    def _build_windows(self) -> List[Tuple[int, List[int], int]]:
        """
        Build sliding window indices for all sequences.

        Logic:
        - 前3个样本(0,1,2)不能作为目标，因为没有足够的历史数据
        - 后3个样本不能作为目标（需要预留下一时刻作为label）
        - 窗口[i-3, i-2, i-1, i]用于构建样本i的数据
        - Label是样本i+1的unit1_beam_index
        - 例如：[0,1,2,3]构建样本3的数据，label是样本4的beam_index

        For sequence with n samples:
        - First valid target: index window_size-1 (e.g., 3 for window_size=4)
        - Last valid target: index n-2-edge_drop (need next sample for label)
        - Window for target i: [i-window_size+1, ..., i]
        - Label: beam_index of sample i+1

        Returns:
            List of (seq_index, [df_indices], label_df_idx) tuples
        """
        windows = []

        # Group by seq_index
        grouped = self.df.groupby('seq_index')

        for seq_idx, group in grouped:
            # Get DataFrame indices for this sequence
            df_indices = group.index.tolist()
            n_samples = len(df_indices)

            # Check if sequence is long enough
            # Need at least window_size + edge_drop + 1 samples
            # (+1 because we need next sample for label)
            min_length = self.window_size + self.edge_drop + 1
            if n_samples < min_length:
                if self.verbose:
                    print(f"  Warning: seq_index={seq_idx} has only {n_samples} samples, "
                          f"need at least {min_length}. Skipping.")
                continue

            # First valid target: need window_size-1 previous samples
            # So first target index = window_size - 1 = 3 (for window_size=4)
            first_target = self.window_size - 1

            # Last valid target: need 1 sample for label + edge_drop samples after
            # So last target index = n_samples - 2 - edge_drop
            last_target = n_samples - 2 - self.edge_drop

            # Generate windows for each valid target
            for target_idx in range(first_target, last_target + 1):
                # Window starts at target_idx - window_size + 1
                start_idx = target_idx - self.window_size + 1
                window_df_indices = df_indices[start_idx:start_idx + self.window_size]
                # Label is the next sample's index
                label_df_idx = df_indices[target_idx + 1]
                windows.append((seq_idx, window_df_indices, label_df_idx))

        return windows

    def _compute_gps_stats(self):
        """Compute mean and std of GPS coordinates from the dataset."""
        if self.verbose:
            print("Computing GPS statistics...")

        all_gps = []

        # Sample for efficiency (use all if small dataset)
        n_samples = min(2000, len(self.df))
        sample_indices = np.random.choice(len(self.df), size=n_samples, replace=False)

        for df_idx in sample_indices:
            gps_path = self._get_full_path(self.df.loc[df_idx, 'unit2_loc_cal'])
            try:
                gps_coords = load_gps(gps_path)
                all_gps.append(gps_coords)
            except Exception:
                continue

        all_gps = np.stack(all_gps, axis=0)  # (N, 2)

        self.gps_mean = all_gps.mean(axis=0).astype(np.float32)
        self.gps_std = all_gps.std(axis=0).astype(np.float32)

        # Avoid division by zero
        self.gps_std = np.maximum(self.gps_std, 1e-8)

    def _print_statistics(self):
        """Print dataset statistics."""
        print("=" * 60)
        print(f"MultiModal Dataset Statistics ({self.split})")
        print("=" * 60)

        # Count samples per sequence
        seq_counts = self.df['seq_index'].value_counts().sort_index()

        window_counts = {}
        for seq_idx, _, _ in self.windows:
            window_counts[seq_idx] = window_counts.get(seq_idx, 0) + 1

        print(f"\nSplit: {self.split} (train_ratio={self.train_ratio})")
        print(f"Total sequences: {len(seq_counts)}")
        print(f"Total raw samples: {len(self.df)}")
        print(f"Total windows: {len(self.windows)}")
        print(f"Window size: {self.window_size}")
        print(f"Edge drop: {self.edge_drop}")

        if self.normalize_gps and self.gps_mean is not None:
            print(f"\nGPS Normalization (Z-score):")
            print(f"  Mean: lat={self.gps_mean[0]:.6f}, lon={self.gps_mean[1]:.6f}")
            print(f"  Std:  lat={self.gps_std[0]:.6f}, lon={self.gps_std[1]:.6f}")

        print("=" * 60)

    def __len__(self) -> int:
        """Return total number of windows."""
        return len(self.windows)

    def _get_full_path(self, relative_path: str) -> str:
        """Convert relative path to full path."""
        if relative_path.startswith('./'):
            relative_path = relative_path[2:]
        return os.path.join(self.root_dir, relative_path)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single window sample.

        Args:
            idx: Index of the window

        Returns:
            Dictionary containing:
                - 'gps': Tensor of shape (4, 2) - GPS coordinates (normalized if enabled)
                - 'lidar': Tensor of shape (4, 216) - LiDAR data
                - 'image': Tensor of shape (3, 540, 960) - RGB image (last frame)
                - 'label': int - Next time step's beam index (unit1_beam_index)
                - 'seq_index': int - Sequence index
                - 'window_indices': List[int] - Original DataFrame indices
        """
        seq_idx, window_df_indices, label_df_idx = self.windows[idx]

        # Load GPS data for all 4 time steps
        gps_list = []
        for df_idx in window_df_indices:
            gps_path = self._get_full_path(self.df.loc[df_idx, 'unit2_loc_cal'])
            gps_coords = load_gps(gps_path)  # (2,)
            gps_list.append(gps_coords)
        gps_data = np.stack(gps_list, axis=0)  # (4, 2)

        # Normalize GPS if enabled
        if self.normalize_gps and self.gps_mean is not None:
            gps_data = (gps_data - self.gps_mean) / self.gps_std

        # Load LiDAR data for all 4 time steps
        lidar_list = []
        for df_idx in window_df_indices:
            lidar_path = self._get_full_path(self.df.loc[df_idx, 'unit1_lidar_SCR'])
            lidar_data = load_lidar(lidar_path)  # (216,)
            lidar_list.append(lidar_data)
        lidar_data = np.stack(lidar_list, axis=0)  # (4, 216)

        # Load image (only the last frame in the window)
        last_df_idx = window_df_indices[-1]
        image_path = self._get_full_path(self.df.loc[last_df_idx, 'unit1_rgb'])
        image_data = load_image(image_path, self.transform)  # (3, 540, 960)

        # Get label: next time step's beam index
        label = int(self.df.loc[label_df_idx, 'unit1_beam_index'])

        return {
            'gps': torch.from_numpy(gps_data.astype(np.float32)),  # (4, 2)
            'lidar': torch.from_numpy(lidar_data),                  # (4, 216)
            'image': image_data,                                    # (3, 540, 960)
            'label': label,                                         # int: next beam index
            'seq_index': seq_idx,
            'window_indices': window_df_indices,
        }

    def get_gps_stats(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Return GPS normalization statistics.

        Returns:
            Dict with 'mean' and 'std' arrays, or None if not normalized
        """
        if self.gps_mean is not None:
            return {'mean': self.gps_mean, 'std': self.gps_std}
        return None

    def get_sequence_info(self) -> Dict[int, int]:
        """Get number of windows per sequence."""
        seq_window_counts = {}
        for seq_idx, _, _ in self.windows:
            seq_window_counts[seq_idx] = seq_window_counts.get(seq_idx, 0) + 1
        return seq_window_counts


def create_dataloaders(
    csv_path: str,
    root_dir: str,
    batch_size: int = 32,
    train_ratio: float = 0.8,
    num_workers: int = 4,
    window_size: int = 4,
    edge_drop: int = 3,
    normalize_gps: bool = True,
    transform: Optional[Callable] = None,
    random_seed: int = 42
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, Optional[Dict]]:
    """
    Create train and test DataLoaders with consistent GPS normalization.

    Args:
        csv_path: Path to CSV index file
        root_dir: Root directory for data
        batch_size: Batch size
        train_ratio: Ratio of sequences for training
        num_workers: Number of workers for data loading
        window_size: Sliding window size
        edge_drop: Samples to drop from sequence edges
        normalize_gps: Whether to normalize GPS coordinates
        transform: Image transform
        random_seed: Random seed for reproducible split

    Returns:
        Tuple of (train_loader, test_loader, gps_stats)
    """
    # Create train dataset first to compute GPS stats
    train_dataset = MultiModalDataset(
        csv_path=csv_path,
        root_dir=root_dir,
        split='train',
        train_ratio=train_ratio,
        window_size=window_size,
        edge_drop=edge_drop,
        transform=transform,
        normalize_gps=normalize_gps,
        gps_stats=None,  # Compute from train data
        random_seed=random_seed
    )

    # Get GPS stats from train set
    gps_stats = train_dataset.get_gps_stats()

    # Create test dataset with train GPS stats
    test_dataset = MultiModalDataset(
        csv_path=csv_path,
        root_dir=root_dir,
        split='test',
        train_ratio=train_ratio,
        window_size=window_size,
        edge_drop=edge_drop,
        transform=transform,
        normalize_gps=normalize_gps,
        gps_stats=gps_stats,  # Use train stats!
        random_seed=random_seed
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader, gps_stats


if __name__ == "__main__":
    print("=" * 60)
    print("MultiModal Dataset Test")
    print("=" * 60)

    # Paths
    csv_path = '/home/changyuan/wireless_context/scenario9_dev/scenario9.csv'
    root_dir = '/home/changyuan/wireless_context/scenario9_dev'

    # Test with percentage split and GPS normalization
    print("\n>>> Creating Training Dataset (80% split, GPS normalized)...")
    train_dataset = MultiModalDataset(
        csv_path=csv_path,
        root_dir=root_dir,
        split='train',
        train_ratio=0.8,
        normalize_gps=True,
        random_seed=42
    )
    print(f"\nTrain samples: {len(train_dataset)}")

    # Get GPS stats for test set
    gps_stats = train_dataset.get_gps_stats()

    # Test first sample
    print("\n>>> Loading first sample...")
    sample = train_dataset[0]
    print(f"GPS shape: {sample['gps'].shape}")        # (4, 2)
    print(f"GPS dtype: {sample['gps'].dtype}")
    print(f"GPS values (normalized):\n{sample['gps']}")
    print(f"GPS range: [{sample['gps'].min():.4f}, {sample['gps'].max():.4f}]")

    print(f"\nLiDAR shape: {sample['lidar'].shape}")  # (4, 216)
    print(f"LiDAR range: [{sample['lidar'].min():.4f}, {sample['lidar'].max():.4f}]")

    print(f"\nImage shape: {sample['image'].shape}")  # (3, 540, 960)
    print(f"Image range: [{sample['image'].min():.4f}, {sample['image'].max():.4f}]")

    print(f"\nLabel (next beam index): {sample['label']}")
    print(f"Seq index: {sample['seq_index']}")
    print(f"Window indices: {sample['window_indices']}")

    # Shape assertions
    assert sample['gps'].shape == (4, 2), f"GPS shape mismatch"
    assert sample['lidar'].shape == (4, 216), f"LiDAR shape mismatch"
    assert sample['image'].shape == (3, 540, 960), f"Image shape mismatch"
    assert isinstance(sample['label'], int), f"Label should be int"
    print("\nShape assertions passed!")

    # Test test set with train stats
    print("\n>>> Creating Test Dataset (using train GPS stats)...")
    test_dataset = MultiModalDataset(
        csv_path=csv_path,
        root_dir=root_dir,
        split='test',
        train_ratio=0.8,
        normalize_gps=True,
        gps_stats=gps_stats,  # Important: use train stats!
        random_seed=42
    )
    print(f"\nTest samples: {len(test_dataset)}")

    # Test DataLoader
    print("\n>>> Testing DataLoader...")
    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0
    )

    batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  GPS:   {batch['gps'].shape}")      # (4, 4, 2)
    print(f"  LiDAR: {batch['lidar'].shape}")    # (4, 4, 216)
    print(f"  Image: {batch['image'].shape}")    # (4, 3, 540, 960)
    print(f"  Label: {batch['label']}")          # (4,) tensor of beam indices

    # Verify GPS is normalized (values should be roughly in [-3, 3] range)
    print(f"\nGPS batch stats (should be ~0 mean, ~1 std):")
    print(f"  Mean: {batch['gps'].mean().item():.4f}")
    print(f"  Std:  {batch['gps'].std().item():.4f}")

    assert batch['gps'].shape == (4, 4, 2), "Batch GPS shape mismatch"
    assert batch['lidar'].shape == (4, 4, 216), "Batch LiDAR shape mismatch"
    assert batch['image'].shape == (4, 3, 540, 960), "Batch Image shape mismatch"
    assert batch['label'].shape == (4,), "Batch Label shape mismatch"

    # Test helper function
    print("\n>>> Testing create_dataloaders helper...")
    train_loader, test_loader, stats = create_dataloaders(
        csv_path=csv_path,
        root_dir=root_dir,
        batch_size=8,
        train_ratio=0.8,
        num_workers=0
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
