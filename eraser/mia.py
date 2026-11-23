"""The audit: a membership-inference attack that tries to tell whether a given point was in a model's training
data at all, from nothing but the model's own behaviour on it. This is what actually *proves* a deletion worked
— a point missing from a lookup table proves nothing about whether the model still carries its influence; a
collapsed attack confidence is evidence about the model itself.

The attack is the standard, simple kind (Yeom et al., 2018; no shadow models): a model tends to have lower loss
on points it trained on than on points it never saw, and that gap is the signal. A one-feature logistic
regression on ``log(loss)`` — calibrated on points *known* to be members (a sample of training data) against
points *known* to be non-members (the held-out set, which no shard ever trains on) — scores any point in
between. (An earlier version added the model's top-class confidence and its prediction entropy as two more
features; for a two-class model the two are a deterministic function of each other, so they are collinear, and
adding both made the fitted attack numerically unstable — occasionally even the *wrong* sign on a held-out point.
Loss alone is simpler and did not have that problem.)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import per_example_loss, predict_proba


def features(model, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """``log(loss)`` rather than raw loss: a confidently-right prediction's loss can be six orders of magnitude
    smaller than a wrong one's, and that heavy tail otherwise swamps the z-score standardisation used to train
    the attack, burying the (real, but much smaller in absolute terms) gap between members and non-members that
    both sit on the "confident" side."""
    loss = per_example_loss(predict_proba(model, X), y)
    return np.log(loss + 1e-9).reshape(-1, 1)


@dataclass
class AttackModel:
    w: np.ndarray       # 3
    b: float
    mean: np.ndarray    # feature standardisation, fit on the calibration data
    std: np.ndarray

    def score(self, model, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Probability the attacker assigns that each row of ``(X, y)`` was in ``model``'s training set."""
        f = (features(model, X, y) - self.mean) / self.std
        z = f @ self.w + self.b
        return 1.0 / (1.0 + np.exp(-z))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def train_attack(model, member_X: np.ndarray, member_y: np.ndarray, nonmember_X: np.ndarray, nonmember_y: np.ndarray,
                  steps: int = 300, lr: float = 0.5, seed: int = 0) -> AttackModel:
    """A logistic regression, trained by hand-derived gradient descent (the same shape of math as the main
    model, at a much smaller scale) to separate ``model``'s known members from known non-members on
    ``features()``."""
    f_mem = features(model, member_X, member_y)
    f_non = features(model, nonmember_X, nonmember_y)
    f = np.concatenate([f_mem, f_non])
    labels = np.concatenate([np.ones(len(f_mem)), np.zeros(len(f_non))])
    mean, std = f.mean(axis=0), f.std(axis=0) + 1e-9
    fz = (f - mean) / std
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 0.1, f.shape[1])
    b = 0.0
    n = len(labels)
    for _ in range(steps):
        p = _sigmoid(fz @ w + b)
        grad = p - labels
        w -= lr * (fz.T @ grad) / n
        b -= lr * grad.mean()
    return AttackModel(w, b, mean, std)


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """The attack's own success rate, measured honestly: the probability a random member scores higher than a
    random non-member (0.5 = the attack is worthless; 1.0 = perfect). Computed exactly via rank statistics, no
    sampling, since the calibration sets here are always small enough for it."""
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    n1 = labels.sum()
    n0 = len(labels) - n1
    if n1 == 0 or n0 == 0:
        return 0.5
    sum_ranks_pos = ranks[labels == 1].sum()
    return float((sum_ranks_pos - n1 * (n1 + 1) / 2) / (n1 * n0))
