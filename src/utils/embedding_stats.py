from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader


def compute_embedding_stats(embeddings: Tensor) -> Dict[str, float]:
    """
    Compute representation collapse diagnostics from a batch of jet embeddings.

    Parameters
    ----------
    embeddings : Tensor, shape (N, D)
        Mean-pooled encoder output across N jets.

    Returns
    -------
    dict with:
        mean_var       — average per-dimension variance (collapse → 0)
        effective_rank — SVD-entropy rank (Roy 2007); collapse → 1, full → D
        mean_cos_sim   — mean pairwise cosine similarity (collapse → 1)
    """
    embeddings = embeddings.float()
    N, D = embeddings.shape

    mean_var = embeddings.var(dim=0).mean().item()

    # Effective rank from normalised singular value entropy
    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    _, S, _ = torch.linalg.svd(centered, full_matrices=False)
    S = S.clamp(min=1e-8)
    p = S / S.sum()
    effective_rank = torch.exp(-(p * p.log()).sum()).item()

    # Mean pairwise cosine similarity (sample for speed)
    n_sample = min(N, 512)
    idx = torch.randperm(N, device=embeddings.device)[:n_sample]
    sample = F.normalize(embeddings[idx], dim=-1)
    cos_mat = sample @ sample.T
    off_diag_mask = ~torch.eye(n_sample, dtype=torch.bool, device=embeddings.device)
    mean_cos_sim = cos_mat[off_diag_mask].mean().item()

    return {
        'mean_var': mean_var,
        'effective_rank': effective_rank,
        'mean_cos_sim': mean_cos_sim,
    }


@torch.no_grad()
def probe_encoder_stats(
    encoder: torch.nn.Module,
    processor: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_jets: int = 4096,
) -> Dict[str, float]:
    """
    Extract mean-pooled jet embeddings from a pretrained encoder and return
    collapse diagnostics.

    Parameters
    ----------
    encoder   : LorentzParTEncoder (already on device, in eval mode)
    processor : ParticleProcessor  (already on device)
    loader    : DataLoader yielding (X, ...) — X shape (B, N, 4)
    device    : torch.device
    max_jets  : int — cap on how many jets to use (speed vs accuracy trade-off)

    Returns
    -------
    dict — output of compute_embedding_stats
    """
    encoder.eval()
    all_embeds = []
    collected = 0

    for batch in loader:
        if collected >= max_jets:
            break

        X = batch[0].to(device)
        padding_mask = (X[..., 3] == 0).float()

        mv, U = processor(X)
        embeddings = encoder(mv, padding_mask, U)  # (B, N, D)

        valid = 1.0 - padding_mask
        valid_sum = valid.sum(dim=1, keepdim=True).clamp(min=1.0)
        jet_embed = (embeddings * valid.unsqueeze(-1)).sum(dim=1) / valid_sum  # (B, D)

        all_embeds.append(jet_embed.cpu())
        collected += X.size(0)

    all_embeds = torch.cat(all_embeds, dim=0)[:max_jets]
    return compute_embedding_stats(all_embeds)
