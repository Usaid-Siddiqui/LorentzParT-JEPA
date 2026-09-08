"""
Streaming JetClass dataset for the Phase 7 (100M) scaling study — no weaver dependency.

Why: at >= 10M jets the in-memory `NpyJetClassDataset` is infeasible (100M x 14 x 128 x f32 ~ 870 GB).
This streams ROOT files. JetClass files are SINGLE-CLASS, so balanced batches require interleaving the
10 class streams — a naive shuffled-file-list + buffer gives class-skewed batches (a memory-safe buffer
is far smaller than a 100k-jet file). We therefore ROUND-ROBIN across the class streams with CHUNKED
reads (`uproot.iterate`, never holding a whole file), feeding a shuffle buffer.

Split of responsibility (M3 in experiments/phase7_scaling/README.md):
  - class grouping + DDP/worker sharding + round-robin scheduling + shuffle  → tested locally (synthetic reader)
  - the actual ROOT chunk reader (`_uproot_chunk_reader`)                    → VALIDATE ON val_5M (no local ROOT)

Returns (particles (max_particles, F) float32, label (10,) float32). 4-vector cols normalized here;
extra features assumed standardized at extraction (as in prepare_data --with-displacement/pid).
"""

import glob
import os
import random
import re
from collections import defaultdict
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

KINEMATIC = ['part_pt', 'part_eta', 'part_phi', 'part_energy']
LABELS = ['label_QCD', 'label_Hbb', 'label_Hcc', 'label_Hgg', 'label_H4q',
          'label_Hqql', 'label_Zqq', 'label_Wqq', 'label_Tbqq', 'label_Tbl']


def _class_key(path: str) -> str:
    """Class = filename with the trailing _<digits>.root stripped (e.g. HToBB_120.root -> HToBB)."""
    return re.sub(r'_\d+\.root$', '', os.path.basename(path))


def _uproot_chunk_reader(path: str, features: List[str], max_particles: int,
                         chunk_size: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield (X (n, F, max_particles), y (n, 10)) chunks from one ROOT file. VALIDATE ON CLUSTER.

    Mirrors dataloader.read_file's feature computation (pt/eta/phi from px/py/pz via `vector`) but
    chunked so only `chunk_size` jets are resident at a time."""
    import awkward as ak
    import uproot
    import vector
    vector.register_awkward()

    def pad(a):
        a = ak.fill_none(ak.pad_none(a, max_particles, clip=True), 0)
        return ak.values_astype(a, 'float32')

    read_branches = ['part_px', 'part_py', 'part_pz', 'part_energy'] + \
                    [f for f in features if f not in KINEMATIC] + LABELS
    for table in uproot.iterate(f"{path}:tree", expressions=read_branches, step_size=chunk_size):
        p4 = vector.zip({'px': table['part_px'], 'py': table['part_py'],
                         'pz': table['part_pz'], 'energy': table['part_energy']})
        derived = {'part_pt': p4.pt, 'part_eta': p4.eta, 'part_phi': p4.phi,
                   'part_energy': table['part_energy']}
        cols = [derived[f] if f in derived else table[f] for f in features]
        X = np.stack([ak.to_numpy(pad(c)) for c in cols], axis=1)          # (n, F, P)
        y = np.stack([ak.to_numpy(table[l]).astype('int') for l in LABELS], axis=1)  # (n, 10)
        yield X.astype(np.float32), y.astype(np.float32)


class StreamingJetClassDataset(IterableDataset):
    def __init__(
        self,
        data_dir: str,
        particle_features: List[str],
        norm_dict: Dict[str, Tuple[float, float]],
        normalize: List[bool] = [True, False, False, True],
        max_particles: int = 128,
        chunk_size: int = 1000,
        shuffle_buffer: int = 20000,
        seed: int = 42,
        chunk_reader: Optional[Callable] = None,   # injectable for testing; default = uproot
    ):
        super().__init__()
        files = sorted(glob.glob(os.path.join(data_dir, '*.root')))
        if not files:
            raise FileNotFoundError(f"no .root files in {data_dir}")
        self.by_class: Dict[str, List[str]] = defaultdict(list)
        for f in files:
            self.by_class[_class_key(f)].append(f)
        self.feats = particle_features
        self.norm_dict = norm_dict
        self.normalize = normalize
        self.max_particles = max_particles
        self.chunk_size = chunk_size
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.epoch = 0
        self._read = chunk_reader or _uproot_chunk_reader
        self._means = np.array([norm_dict[k][0] for k in ['pT', 'eta', 'phi', 'energy']], np.float32)
        self._stds = np.array([1.0, norm_dict['eta'][1], norm_dict['phi'][1], 1.0], np.float32)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def _consumer(self) -> Tuple[int, int]:
        """(index, count) over all DDP-rank x DataLoader-worker consumers."""
        rank, world = (0, 1)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank, world = torch.distributed.get_rank(), torch.distributed.get_world_size()
        wi = get_worker_info()
        worker, nworkers = (wi.id, wi.num_workers) if wi else (0, 1)
        return rank * nworkers + worker, world * nworkers

    def _norm(self, part: np.ndarray) -> np.ndarray:
        for i in range(4):
            if not self.normalize[i]:
                continue
            if i in (0, 3):
                part[:, i] = part[:, i] / self._means[i]
            else:
                part[:, i] = (part[:, i] - self._means[i]) / self._stds[i]
        return part

    def _class_chunk_stream(self, files: List[str], rng: random.Random):
        """Chain the (shuffled) files of one class into a stream of chunks."""
        files = list(files)
        rng.shuffle(files)
        for fp in files:
            yield from self._read(fp, self.feats, self.max_particles, self.chunk_size)

    def __iter__(self):
        cidx, ncons = self._consumer()
        rng = random.Random(self.seed + self.epoch)
        crng = random.Random(self.seed + self.epoch + 7919 * cidx)

        # Shard each class's files across consumers → each consumer sees all classes, balanced.
        streams = []
        for cls in sorted(self.by_class):
            my = self.by_class[cls][cidx::ncons]
            if my:
                streams.append(self._class_chunk_stream(my, crng))

        buf: List[Tuple[torch.Tensor, torch.Tensor]] = []
        active = list(range(len(streams)))
        while active:
            still = []
            for s in active:                       # one chunk per class per round → balanced
                try:
                    X, y = next(streams[s])
                except StopIteration:
                    continue
                for j in range(len(X)):
                    part = X[j].T.copy()            # (P, F)
                    self._norm(part)
                    item = (torch.from_numpy(part), torch.from_numpy(y[j]))
                    if len(buf) < self.shuffle_buffer:
                        buf.append(item)
                    else:
                        k = crng.randrange(len(buf)); yield buf[k]; buf[k] = item
                still.append(s)
            active = still
        crng.shuffle(buf)
        yield from buf


if __name__ == '__main__':
    # Validate the ROOT reader + balance on real data (cluster):
    #   python -m src.utils.data.streaming_jetclass /path/to/val_5M
    import sys
    from collections import Counter
    NORM = {'pT': (92.73, 105.84), 'eta': (0.00057, 0.9175),
            'phi': (-0.00041, 1.8137), 'energy': (133.87, 167.53)}
    ds = StreamingJetClassDataset(sys.argv[1], particle_features=KINEMATIC, norm_dict=NORM,
                                  chunk_size=1000, shuffle_buffer=20000)
    labels, shape = Counter(), None
    for i, (x, y) in enumerate(ds):
        shape = tuple(x.shape); labels[int(y.argmax())] += 1
        if i + 1 >= 50000:
            break
    print(f"sample shape {shape} · {sum(labels.values())} jets")
    print("class balance over first 50k jets (want ~5000 each):")
    for c in range(10):
        print(f"  class {c}: {labels.get(c, 0)}")
