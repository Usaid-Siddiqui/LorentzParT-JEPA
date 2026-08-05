"""
Phase 4 downstream metrics. Classification AUC + accuracy for any k, plus the
HEP tagger metric — background rejection 1/ε_B at fixed signal efficiency ε_S=0.5 —
for binary tasks.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def background_rejection(y_true_bin, y_score, eps_s=0.5):
    """1/ε_B at signal efficiency ε_S (default 0.5). y_true_bin: 1=signal, 0=bkg.

    A near-perfect classifier drives ε_B→0, so 1/ε_B blows up (and jitters wildly
    seed-to-seed from ROC-interpolation noise). Cap it at the statistical limit: with
    N_bkg background test jets you cannot resolve a rejection finer than ~N_bkg, so
    floor ε_B at 1/N_bkg — the reported value saturates at N_bkg ("test-stat limited")
    instead of exploding to 1e9.
    """
    fpr, tpr, _ = roc_curve(y_true_bin, y_score)     # fpr = ε_B, tpr = ε_S
    eps_b = float(np.interp(eps_s, tpr, fpr))         # background eff at the target signal eff
    n_bkg = int((y_true_bin == 0).sum())
    eps_b_floor = 1.0 / max(n_bkg, 1)
    return 1.0 / max(eps_b, eps_b_floor)


def classification_metrics(y_true_onehot, y_prob, kind, signal_idx=None):
    """
    Parameters
    ----------
    y_true_onehot : (N, k) one-hot ground truth
    y_prob        : (N, k) predicted probabilities (softmax)
    kind          : 'binary' or 'multiclass'
    signal_idx    : remapped index of the signal class (binary only) for 1/ε_B

    Returns a dict with test_acc, test_auc, and (binary) bkg_rej_at_0.5 + per-class.
    """
    y_true = y_true_onehot.argmax(1)
    k = y_prob.shape[1]
    acc = float((y_prob.argmax(1) == y_true).mean())

    if kind == 'binary':
        sig = signal_idx if signal_idx is not None else 1
        y_bin = (y_true == sig).astype(int)
        score = y_prob[:, sig]
        auc = float(roc_auc_score(y_bin, score))       # threshold-symmetric for binary
        return {
            'test_acc': acc,
            'test_auc': auc,
            'bkg_rej_at_0.5': float(background_rejection(y_bin, score, 0.5)),
            'signal_idx': int(sig),
        }

    # multiclass: OVO macro AUC + per-class one-vs-rest AUC/acc
    auc = float(roc_auc_score(y_true_onehot, y_prob, average='macro', multi_class='ovo'))
    per_class_auc, per_class_acc = [], []
    for i in range(k):
        yi = (y_true == i).astype(int)
        per_class_auc.append(float(roc_auc_score(yi, y_prob[:, i])))
        mask = y_true == i
        per_class_acc.append(float((y_prob[mask].argmax(1) == i).mean()) if mask.sum() else 0.0)
    return {
        'test_acc': acc,
        'test_auc': auc,
        'per_class_auc': per_class_auc,
        'per_class_acc': per_class_acc,
    }
