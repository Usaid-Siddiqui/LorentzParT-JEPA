"""
Phase 4 downstream tasks — fine-grained decay discrimination by RELABELING the
existing 10-class JetClass arrays (no re-extraction). Each task keeps the jets whose
argmax label is in its class set and remaps those labels to 0..k-1.

10-class index reference:
  0 QCD/Zνν · 1 H→bb · 2 H→cc · 3 H→gg · 4 H→4q · 5 H→lνqq · 6 Z→qq · 7 W→qq · 8 t→bqq · 9 t→blν
"""

import numpy as np
from torch.utils.data import Dataset

from src.utils.data import NpyJetClassDataset


# name -> {classes (original 10-class indices), kind, signal (original index, binary only)}
TASKS = {
    'top_chan':  {'classes': [8, 9],          'kind': 'binary',     'signal': 8},  # hadronic vs semileptonic top
    'hbb_hcc':   {'classes': [1, 2],          'kind': 'binary',     'signal': 1},  # H→bb vs H→cc
    'wz':        {'classes': [6, 7],          'kind': 'binary',     'signal': 7},  # W→qq vs Z→qq (W = signal)
    'higgs_had': {'classes': [1, 2, 3],       'kind': 'multiclass'},               # H→ bb/cc/gg
    'higgs5':    {'classes': [1, 2, 3, 4, 5], 'kind': 'multiclass'},               # H→ bb/cc/gg/4q/lνqq
}


def signal_index(task: str) -> int:
    """Remapped (0..k-1) index of the signal class for a binary task."""
    t = TASKS[task]
    return sorted(t['classes']).index(t['signal'])


class TaskDataset(NpyJetClassDataset):
    """
    A relabeled view of the 10-class arrays for one downstream task.

    Keeps only jets whose argmax label is in ``classes`` and remaps to a k-dim
    one-hot (classes sorted → 0..k-1), so the parent ``__getitem__`` returns
    ``(particles (128,4), label (k,))`` unchanged. Loads only the selected jets
    via a memmap (never the full array), and subsamples BALANCED to ``max_samples``
    with a fixed ``subsample_seed`` so every encoder/protocol/training-seed on a
    task sees the identical subset (fair transfer, reproducible, resumable).
    """

    def __init__(
        self,
        particles_path: str,
        labels_path: str,
        classes,
        normalize=[True, False, False, True],
        norm_dict=None,
        max_samples=None,
        subsample_seed: int = 0,
    ):
        Dataset.__init__(self)                       # skip parent's full-array load
        classes = sorted(classes)
        k = len(classes)

        labels = np.load(labels_path)                # (N, 10) one-hot — small
        lab10 = labels.argmax(1)
        rng = np.random.default_rng(subsample_seed)

        per = None if max_samples is None else max_samples // k
        sel = []
        for c in classes:
            idx_c = np.where(lab10 == c)[0]
            if per is not None and len(idx_c) > per:
                idx_c = rng.choice(idx_c, size=per, replace=False)
            sel.append(idx_c)
        sel = np.sort(np.concatenate(sel))

        Xmm = np.load(particles_path, mmap_mode='r')  # (N, 4, 128) — materialize only selected
        self.X_particles = np.ascontiguousarray(Xmm[sel])

        remap = {c: i for i, c in enumerate(classes)}
        new = np.array([remap[int(c)] for c in lab10[sel]], dtype=np.int64)
        self.y = np.eye(k, dtype=np.float32)[new]     # remapped k-dim one-hot

        # normalization setup — mirror NpyJetClassDataset.__init__
        self.normalize = normalize
        self.norm_dict = norm_dict
        self.mask_mode = None
        self.num_mask = 1
        self._feat_names = ['pT', 'eta', 'phi', 'energy']
        if norm_dict is not None:
            means, stds = [], []
            for i, key in enumerate(self._feat_names):
                m, s = norm_dict[key]
                means.append(m)
                stds.append(s if i in (1, 2) else 1.0)
            self._means = np.array(means, dtype=np.float32)
            self._stds = np.array(stds, dtype=np.float32)
            self._use = np.array(self.normalize, dtype=bool)

        self.classes = classes
        self.k = k
