import os
from typing import List, Tuple, Dict, Optional, OrderedDict

import numpy as np

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .dataloader import read_file


class NpyJetClassDataset(Dataset):
    """
    In-memory dataset that loads pre-processed JetClass numpy arrays.

    Designed for the 100k demo subset created by scripts/prepare_data.py.
    Particle arrays are stored as (N, 4, max_particles); this class transposes
    them to (max_particles, 4) per jet so they match the format expected by the
    ParticleProcessor and LorentzParT model.

    Parameters
    ----------
    particles_path : str
        Path to ``particles.npy``, shape ``(N, 4, 128)``.
    labels_path : str
        Path to ``labels.npy``, shape ``(N, 10)``.
    normalize : list of bool
        Which features [pT, eta, phi, energy] to normalise.
    norm_dict : dict, optional
        ``{feature: (mean, std)}`` for normalisation.
    mask_mode : str, optional
        ``'random'``, ``'biased'``, ``'first'`` or ``None``.
        When set, __getitem__ returns ``(masked_particles, masked_target, mask_idx)``.
        When ``None``, returns ``(particles, label)``.
    num_mask : int, optional
        Number of particles to mask per jet (default 1). Only used when
        ``mask_mode`` is set. For K>1, particles are sampled without replacement.
        ``biased`` mode applies biased sampling for the first particle only;
        remaining K-1 are sampled randomly.
    """

    def __init__(
        self,
        particles_path: str,
        labels_path: str,
        normalize: List[bool] = [True, False, False, True],
        norm_dict: Optional[Dict[str, Tuple[float, float]]] = None,
        mask_mode: Optional[str] = None,
        num_mask: int = 1,
    ):
        super().__init__()
        self.X_particles = np.load(particles_path)   # (N, 4, 128)
        self.y = np.load(labels_path).astype(np.float32)
        self.normalize = normalize
        self.norm_dict = norm_dict
        self.mask_mode = mask_mode
        self.num_mask = num_mask

        # Pre-build norm arrays for fast broadcasting
        self._feat_names = ['pT', 'eta', 'phi', 'energy']
        if self.norm_dict is not None:
            means, stds = [], []
            for i, k in enumerate(self._feat_names):
                m, s = self.norm_dict[k]
                means.append(m)
                stds.append(s if i in (1, 2) else 1.0)
            self._means = np.array(means, dtype=np.float32)
            self._stds = np.array(stds, dtype=np.float32)
            self._use = np.array(self.normalize, dtype=bool)

    def __len__(self) -> int:
        return len(self.X_particles)

    def _apply_norm(self, arr: np.ndarray) -> np.ndarray:
        """Normalise a (max_particles, 4) array in-place and return it."""
        if self.norm_dict is None:
            return arr
        for i in range(4):
            if not self._use[i]:
                continue
            if i in (0, 3):        # pT / energy: divide by mean
                arr[:, i] = arr[:, i] / self._means[i]
            else:                   # eta / phi: (x - mean) / std
                arr[:, i] = (arr[:, i] - self._means[i]) / self._stds[i]
        return arr

    def _mask_particle(
        self, particles: np.ndarray, mode: str = 'random', num_mask: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        valid_idx = np.where(np.any(particles != 0, axis=1))[0]

        if mode == 'first':
            mask_idx = valid_idx[:num_mask]
        elif mode == 'biased' and num_mask == 1:
            # Rejection sampling biased toward low indices (high-pT particles).
            # Kept as-is so existing K=1 biased runs stay bit-reproducible.
            total = np.sum(1 / (np.arange(particles.shape[0]) + 1))
            idx = 127
            u, w = 0, 1
            while (u < w) or (idx not in valid_idx):
                u = np.random.uniform()
                idx = np.random.randint(0, particles.shape[0])
                w = (1 / (idx + 1)) / total
            mask_idx = np.array([idx])
        else:
            # random (uniform) or biased K>1 (high-pT weighted, w ∝ 1/(index+1)).
            # Deficit handling: jets with fewer than num_mask valid particles pad
            # the remainder by repeating, keeping a fixed (K,) shape for batching.
            p = None
            if mode == 'biased':
                w = 1.0 / (valid_idx + 1.0)
                p = w / w.sum()
            n = len(valid_idx)
            if n >= num_mask:
                mask_idx = np.random.choice(valid_idx, size=num_mask, replace=False, p=p)
            else:
                extra = np.random.choice(valid_idx, size=num_mask - n, replace=True, p=p)
                mask_idx = np.concatenate([valid_idx, extra])

        masked_particles = particles.copy()
        masked_targets = masked_particles[mask_idx, :].copy()
        masked_particles[mask_idx, :] = 0.0
        return masked_particles, masked_targets, mask_idx   # mask_idx shape: (K,)

    def __getitem__(self, idx: int) -> Tuple[Tensor, ...]:
        part = self.X_particles[idx].T.copy()   # (128, 4)
        label = self.y[idx]

        if self.mask_mode is not None:
            masked_particles, masked_targets, mask_idx = self._mask_particle(
                part, self.mask_mode, self.num_mask
            )
            self._apply_norm(masked_particles)
            self._apply_norm(masked_targets)
            # masked_targets: (K, 4) for K>1, squeeze to (4,) for K=1 (MAE compat)
            return (
                torch.from_numpy(masked_particles).float(),
                torch.from_numpy(masked_targets.squeeze(0)).float(),
                torch.tensor(mask_idx, dtype=torch.int64),  # (K,)
            )

        self._apply_norm(part)
        return torch.from_numpy(part).float(), torch.from_numpy(label).float()


class JetClassDataset(Dataset):
    """
    A PyTorch dataset class to use JetClass for classification and self-supervised learning.

    Parameters
    ----------
    X_particles: np.ndarray
        The input particle features.
    y: np.ndarray
        The target labels.
    normalize: List[bool], optional
        A list indicating which features to normalize. Default is `[True, False, False, True]` (normalize pT and energy).
    norm_dict: Dict[str, Tuple[float, float]], optional
        A dictionary containing the normalization parameters (mean, std) for each feature.
    mask_mode: str, optional
        The masking mode to use. If not specified, no masking is applied.

    Returns
    -------
    Tuple[Tensor, ...]
        For self-supervised learning: a tuple containing the following elements:
        - The masked particle features tensor of shape (max_num_particles, num_particle_features).
        - The masked target labels tensor of shape (num_particle_features,).
        - The mask index tensor of shape (1,).
        
        For classification: a tuple containing the following elements:
        - The particle features tensor of shape (max_num_particles, num_particle_features).
        - The target labels tensor of shape (num_classes,).

    .. References::
        Huilin Qu, Congqiao Li, and Sitian Qian.
        [Particle Transformer for Jet Tagging](https://arxiv.org/abs/2202.03772).
        In *Proceedings of the 39th International Conference on Machine Learning*, pages 18281-18292, 2022.

        Huilin Qu, Congqiao Li, and Sitian Qian.
        [JetClass: A Large-Scale Dataset for Deep Learning in Jet Physics](https://zenodo.org/records/6619768).
        *Zenodo*, 2022.
    """
    def __init__(
        self,
        X_particles: np.ndarray,
        y: np.ndarray,
        normalize: List[bool] = [True, False, False, True],  # [pT, eta, phi, energy]
        norm_dict: Dict[str, Tuple[float, float]] = None,
        mask_mode: str = None
    ):
        super().__init__()
        self.X_particles = X_particles
        self.y = y
        self.normalize = normalize
        self.norm_dict = norm_dict
        self.mask_mode = mask_mode

    def __len__(self) -> int:
        return len(self.X_particles)

    def __getitem__(self, idx: int) -> Tuple[Tensor, ...]:
        particles = self.X_particles[idx].T  # (max_num_particles, num_particle_features)

        if self.mask_mode is not None:
            particles = particles.copy()
            masked_particles, masked_targets, mask_idx = self._mask_particle(particles, self.mask_mode)

            # Normalize features (in-place)
            feature_names = ['pT', 'eta', 'phi', 'energy']

            if self.norm_dict is not None:
                for i, feature in enumerate(feature_names):
                    if self.normalize[i]:
                        mean, std = self.norm_dict[feature]

                        if i == 0 or i == 3:  # pT or energy where values are strictly positive
                            masked_particles[:, i] = masked_particles[:, i] / mean
                            masked_targets[:, i] = masked_targets[:, i] / mean
                        else:
                            masked_particles[:, i] = (masked_particles[:, i] - mean) / std
                            masked_targets[:, i] = (masked_targets[:, i] - mean) / std

            return (
                torch.tensor(masked_particles, dtype=torch.float32),  # (max_num_particles, num_particle_features)
                torch.tensor(masked_targets.squeeze(0), dtype=torch.float32),  # (num_particle_features,)
                torch.tensor(mask_idx, dtype=torch.int64)  # (1,)
            )
        else:
            # Normalize features (in-place)
            feature_names = ['pT', 'eta', 'phi', 'energy']
            if self.norm_dict is not None:
                for i, feature in enumerate(feature_names):
                    if self.normalize[i]:
                        mean, std = self.norm_dict[feature]
                        if i == 0 or i == 3:  # pT or energy where values are strictly positive
                            particles[:, i] = particles[:, i] / mean
                        else:
                            particles[:, i] = (particles[:, i] - mean) / std

            tensor = torch.from_numpy(particles).float()  # (max_num_particles, num_particle_features)
            label = torch.from_numpy(self.y[idx]).float()  # (num_classes,)

            return tensor, label
        
    def _mask_particle(self, particles: np.ndarray, mode: str = 'random') -> Tuple[np.ndarray, np.ndarray, int]:
        valid_idx = np.where(np.any(particles != 0, axis=1))[0]
        
        if mode == 'random':
            mask_idx = np.array([np.random.choice(valid_idx)])
        elif mode == 'biased':
            total = np.sum(1 / (np.arange(0, particles.shape[0]) + 1))
            mask_idx = 127
            u, w = 0, 1

            while (u < w) or (mask_idx not in valid_idx):
                u = np.random.uniform(0, 1)
                mask_idx = np.random.randint(0, particles.shape[0])
                w = (1 / (mask_idx + 1)) / total
                
            mask_idx = np.array([mask_idx])
        elif mode == 'first':
            mask_idx = valid_idx[:1]
        
        masked_particles = particles.copy()
        masked_targets = masked_particles[mask_idx, :].copy()
        masked_particles[mask_idx, :] = 0.0

        return masked_particles, masked_targets, mask_idx
    

class LazyJetClassDataset(JetClassDataset):
    """
    A PyTorch dataset class to use JetClass for classification and self-supervised learning.
    This class loads data from memory-mapped files on-the-fly to handle large datasets.

    Parameters
    ----------
    data_dir: str
        The directory containing the ROOT files.
    normalize: List[bool], optional
        A list indicating which features to normalize. Default is `[True, False, False, True]` (normalize pT and energy).
    norm_dict: Dict[str, Tuple[float, float]], optional
        A dictionary containing the normalization parameters (mean, std) for each feature.
    mask_mode: str, optional
        The masking mode to use. If not specified, no masking is applied.
    cache_size: int, optional
        The number of files to cache in memory. Default is 10.

    Returns
    -------
    Tuple[Tensor, ...]
        For self-supervised learning: a tuple containing the following elements:
        - The masked particle features tensor of shape (max_num_particles, num_particle_features).
        - The masked target labels tensor of shape (num_particle_features,).
        - The mask index tensor of shape (1,).
        
        For classification: a tuple containing the following elements:
        - The particle features tensor of shape (max_num_particles, num_particle_features).
        - The target labels tensor of shape (num_classes,).

    .. References::
        Huilin Qu, Congqiao Li, and Sitian Qian.
        [Particle Transformer for Jet Tagging](https://arxiv.org/abs/2202.03772).
        In *Proceedings of the 39th International Conference on Machine Learning*, pages 18281-18292, 2022.

        Huilin Qu, Congqiao Li, and Sitian Qian.
        [JetClass: A Large-Scale Dataset for Deep Learning in Jet Physics](https://zenodo.org/records/6619768).
        *Zenodo*, 2022.
    """
    def __init__(
        self,
        data_dir: str,
        normalize: List[bool] = [True, False, False, True],  # [pT, eta, phi, energy]
        norm_dict: Dict[str, Tuple[float, float]] = None,
        mask_mode: str = None,
        cache_size: int = 10
    ):
        super().__init__(None, None, normalize, norm_dict, mask_mode)
        # Sorted absolute file paths; lexicographic groups align with classes
        self.files = sorted(
            os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.root')
        )
        # Build contiguous blocks per class
        files_per_class = len(self.files) // 10
        self.files_by_class = [
            list(range(i * files_per_class, (i + 1) * files_per_class)) for i in range(10)
        ]
        self.events_per_file = 100_000

        self.normalize = normalize
        self.norm_dict = norm_dict
        self.mask_mode = mask_mode

        # Small multi-file LRU cache
        self._cache_size = int(cache_size)
        self._cache = OrderedDict()

        # Pre-build normalization arrays for fast broadcasting
        self._feat_names = ['pT', 'eta', 'phi', 'energy']
        if self.norm_dict is not None:
            means, stds = [], []
            for i, k in enumerate(self._feat_names):
                m, s = self.norm_dict[k]
                # pT / energy: scale by mean only; eta/phi: (x-mean)/std
                means.append(m)
                stds.append(s if i in (1, 2) else 1.0)  # no std use for pT/energy branch

            self._means = np.array(means, dtype=np.float32)
            self._stds = np.array(stds, dtype=np.float32)
            self._use = np.array(self.normalize, dtype=bool)

    def __len__(self) -> int:
        return len(self.files) * self.events_per_file

    def _get_file(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        # LRU hit
        if idx in self._cache:
            particles, labels = self._cache.pop(idx)
            self._cache[idx] = (particles, labels)  # move to end (most-recent)

            return particles, labels

        # Miss → load
        particles, _, labels = read_file(self.files[idx])

        # Insert & evict
        self._cache[idx] = (particles, labels)
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)  # evict least-recent

        return particles, labels

    def _apply_norm_inplace(self, arr: np.ndarray) -> None:
        if self.norm_dict is None:
            return
        
        # pT/energy: x /= mean ; eta/phi: (x-mean)/std
        # Mask off untouched features
        for i in range(4):
            if not self._use[i]:
                continue
            if i in (0, 3):
                arr[:, i] = arr[:, i] / self._means[i]
            else:
                arr[:, i] = (arr[:, i] - self._means[i]) / self._stds[i]

    def _mask_particle(self, particles: np.ndarray, mode: str = 'random'):
        # same logic as your current implementation; omitted for brevity
        valid_idx = np.where(np.any(particles != 0, axis=1))[0]
        if mode == 'random':
            mask_idx = np.array([np.random.choice(valid_idx)])
        elif mode == 'biased':
            total = np.sum(1 / (np.arange(0, particles.shape[0]) + 1))
            mask_idx = 127
            u, w = 0, 1
            while (u < w) or (mask_idx not in valid_idx):
                u = np.random.uniform(0, 1)
                mask_idx = np.random.randint(0, particles.shape[0])
                w = (1 / (mask_idx + 1)) / total

            mask_idx = np.array([mask_idx])
        elif mode == 'first':
            mask_idx = valid_idx[:1]
        else:
            mask_idx = np.array([np.random.choice(valid_idx)])

        masked_particles = particles.copy()
        masked_targets = masked_particles[mask_idx, :].copy()
        masked_particles[mask_idx, :] = 0.0

        return masked_particles, masked_targets, mask_idx

    def __getitem__(self, key: Tuple[int, int]):
        file_idx, event_idx = key
        particles, labels = self._get_file(file_idx)

        # Always copy before any in-place ops (protect cache)
        part = particles[event_idx].T.copy()  # (max_num_particles, 4)
        label = labels[event_idx]
        if self.mask_mode is not None:
            masked_particles, masked_targets, mask_idx = self._mask_particle(part, self.mask_mode)
            # Normalize masked views
            self._apply_norm_inplace(masked_particles)
            self._apply_norm_inplace(masked_targets)

            return (
                torch.tensor(masked_particles, dtype=torch.float32),
                torch.tensor(masked_targets.squeeze(0), dtype=torch.float32),
                torch.tensor(mask_idx, dtype=torch.int64),
            )

        # Classification path
        self._apply_norm_inplace(part)
        
        return torch.from_numpy(part).float(), torch.from_numpy(label).float()
    # def __init__(
    #     self,
    #     data_dir: str,
    #     normalize: List[bool] = [True, False, False, True],  # [pT, eta, phi, energy]
    #     norm_dict: Dict[str, Tuple[float, float]] = None,
    #     mask_mode: str = None,
    #     cache_size: int = 10
    # ):
    #     # Sorted list of absolute file paths
    #     self.files = sorted([
    #         os.path.join(data_dir, fname)
    #         for fname in os.listdir(data_dir)
    #         if fname.endswith('.root')
    #     ])

    #     # Group file indices by class: file index mod 10 gives the class
    #     files_per_class = len(self.files) // 10
    #     self.files_by_class = [
    #         list(range(i * files_per_class, (i + 1) * files_per_class))
    #         for i in range(10)
    #     ]
    #     self.events_per_file = 100_000
    #     self.normalize = normalize
    #     self.norm_dict = norm_dict
    #     self.mask_mode = mask_mode

    #     # LRU cache: at most one loaded file kept in memory
    #     self._cache_file_idx = None
    #     self._cache_particles = None
    #     self._cache_labels = None

    # def _load_file(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
    #     if self._cache_file_idx == idx:
    #         return self._cache_particles, self._cache_labels
        
    #     particles, _, labels = read_file(self.files[idx])
    #     self._cache_file_idx = idx
    #     self._cache_particles = particles
    #     self._cache_labels = labels

    #     return particles, labels

    # def __len__(self) -> int:
    #     return len(self.files) * self.events_per_file

    # def __getitem__(self, key: Tuple[int, int]) -> Tuple[Tensor, ...]:
    #     file_idx, event_idx = key
    #     particles, labels = self._load_file(file_idx)
    #     part = particles[event_idx].T  # (max_num_particles, num_particle_features)
    #     label = labels[event_idx]

    #     part = part.copy()  # to avoid modifying the cached data
    #     if self.mask_mode is not None:
    #         masked_particles, masked_targets, mask_idx = self._mask_particle(part, self.mask_mode)

    #         # Normalize features (in-place)
    #         feature_names = ['pT', 'eta', 'phi', 'energy']

    #         if self.norm_dict is not None:
    #             for i, feature in enumerate(feature_names):
    #                 if self.normalize[i]:
    #                     mean, std = self.norm_dict[feature]

    #                     if i == 0 or i == 3:  # pT or energy where values are strictly positive
    #                         masked_particles[:, i] = masked_particles[:, i] / mean
    #                         masked_targets[:, i] = masked_targets[:, i] / mean
    #                     else:
    #                         masked_particles[:, i] = (masked_particles[:, i] - mean) / std
    #                         masked_targets[:, i] = (masked_targets[:, i] - mean) / std

    #         return (
    #             torch.tensor(masked_particles, dtype=torch.float32),  # (max_num_particles, num_particle_features)
    #             torch.tensor(masked_targets.squeeze(0), dtype=torch.float32),  # (num_particle_features,)
    #             torch.tensor(mask_idx, dtype=torch.int64)  # (1,)
    #         )
    #     else:
    #         # Normalize features (in-place)
    #         if self.norm_dict:
    #             feature_names = ['pT', 'eta', 'phi', 'energy']
    #             for i, feature in enumerate(feature_names):
    #                 if self.normalize[i]:
    #                     mean, std = self.norm_dict[feature]
    #                     if i == 0 or i == 3:  # pT or energy where values are strictly positive
    #                         part[:, i] = part[:, i] / mean
    #                     else:
    #                         part[:, i] = (part[:, i] - mean) / std

    #         part = torch.from_numpy(part).float()
    #         label = torch.from_numpy(label).float()

    #         return part, label