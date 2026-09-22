from typing import Dict, Tuple

import numpy as np


def compute_norm_stats(X_particles: np.ndarray) -> Dict[str, Tuple[float, float]]:
    """Per-feature mean/std of the 4-vector over all NON-PADDED particles.

    ``X_particles`` is (n_jets, n_features, n_particles) — the layout ``JetClassDataset``
    stores (it transposes per item). Features are ordered (pT, eta, phi, energy, ...extras).

    Phase 8: the previous version reshaped by ``shape[2]`` (the PARTICLE count) instead of the
    feature count. It only picked out the right columns because 128 % 4 == 0 made the stride
    happen to align, it used ~4% of the particles, and that 4% was dominated by particle 0 —
    the LEADING particle — so the reported pT mean was ~6x the true one (50.4 vs 8.6 on a
    controlled test). With a feature count that does not divide the particle count (e.g. 14)
    the columns mixed different features outright.
    """
    n_feat = X_particles.shape[1]
    Xp = X_particles.transpose(0, 2, 1).reshape(-1, n_feat)   # (n_jets * n_particles, n_feat)
    Xp = Xp[Xp[:, 0] != 0]                                    # drop padding (pT == 0)

    pT_mean, pT_std = Xp[:, 0].mean(), Xp[:, 0].std()
    eta_mean, eta_std = Xp[:, 1].mean(), Xp[:, 1].std()
    phi_mean, phi_std = Xp[:, 2].mean(), Xp[:, 2].std()
    E_mean, E_std = Xp[:, 3].mean(), Xp[:, 3].std()

    print(f"pt_mean: {pT_mean}, pt_std: {pT_std}")
    print(f"eta_mean: {eta_mean}, eta_std: {eta_std}")
    print(f"phi_mean: {phi_mean}, phi_std: {phi_std}")
    print(f"E_mean: {E_mean}, E_std: {E_std}")
    print(f"(computed over {len(Xp)} valid particles)")

    return {
        'pT': (float(pT_mean), float(pT_std)),
        'eta': (float(eta_mean), float(eta_std)),
        'phi': (float(phi_mean), float(phi_std)),
        'energy': (float(E_mean), float(E_std)),
    }
