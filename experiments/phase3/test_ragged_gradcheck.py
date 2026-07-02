"""
Float64 gradcheck for RaggedInteractionEmbedding (Phase 3).

torch.autograd.gradcheck perturbs each input element by a tiny eps, builds the
finite-difference Jacobian, and compares it against autograd's analytical
Jacobian. Passing means gradients flow CORRECTLY (not just finitely) through the
ragged gather -> BN+MLP -> scatter path — catching the failure modes the
reorganization risks (broken in-place scatter, accidental detach, bad indexing).
Stronger than a finiteness check.

This is NOT an output-parity test vs the stock InteractionEmbedding: ragged is a
deliberate semantic change (BatchNorm over valid pairs only), so its output
differs by design. gradcheck only verifies the ragged module is internally
gradient-correct for its own forward.

    python experiments/phase3/test_ragged_gradcheck.py     # CPU float64 is fine
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.processor import RaggedInteractionEmbedding


def _valid_pairs(B, N, keep_prob=0.7, seed=0):
    """Random valid-particle mask -> (B, N, N) valid-pair mask, with >=1 valid
    particle per jet so BatchNorm always sees enough samples."""
    g = torch.Generator().manual_seed(seed)
    part = torch.rand(B, N, generator=g) < keep_prob
    part[:, 0] = True
    return part[:, :, None] & part[:, None, :]


def gradcheck_wrt_input():
    """Grad w.r.t. U — exercises the gather/scatter data path (the risky part)."""
    torch.manual_seed(0)
    B, N, F = 2, 6, 4
    mod = RaggedInteractionEmbedding(F, [8, 8, 4]).double().train()  # tiny MLP -> fast gradcheck
    vp = _valid_pairs(B, N)
    U = torch.randn(B, N, N, F, dtype=torch.double, requires_grad=True)
    ok = torch.autograd.gradcheck(
        lambda u: mod(u, vp), (U,),
        eps=1e-6, atol=1e-4, rtol=1e-3, nondet_tol=1e-6,
    )
    print("gradcheck w.r.t. U (gather/scatter path):", "PASS" if ok else "FAIL")
    return ok


def gradcheck_wrt_params():
    """Grad w.r.t. the first Linear weight — confirms grads reach the MLP params
    through the scatter, not just the input. Uses functional_call to swap the weight
    WITHOUT detaching it from the gradcheck input (a plain nn.Parameter wrap would
    detach and give an all-zero analytical Jacobian). eval() so BatchNorm uses fixed
    buffers (no in-place buffer mutation across gradcheck's repeated forwards)."""
    from torch.func import functional_call

    torch.manual_seed(1)
    B, N, F = 2, 6, 4
    mod = RaggedInteractionEmbedding(F, [8, 8, 4]).double().eval()
    vp = _valid_pairs(B, N, seed=1)
    U = torch.randn(B, N, N, F, dtype=torch.double)

    params, buffers = dict(mod.named_parameters()), dict(mod.named_buffers())
    name = next(n for n, p in params.items() if p.dim() == 3)   # first Conv1d(k=1) weight
    w = params[name].detach().clone().requires_grad_(True)

    def run(weight):
        return functional_call(mod, {**params, name: weight, **buffers}, (U, vp))

    ok = torch.autograd.gradcheck(run, (w,), eps=1e-6, atol=1e-4, rtol=1e-3, nondet_tol=1e-6)
    print(f"gradcheck w.r.t. {name} (MLP params):", "PASS" if ok else "FAIL")
    return ok


def main():
    results = [gradcheck_wrt_input(), gradcheck_wrt_params()]
    ok = all(results)
    print("\nALL PASS" if ok else "\nFAILED")
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
