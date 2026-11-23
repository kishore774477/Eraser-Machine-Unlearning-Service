"""A small two-layer MLP (ReLU hidden layer, softmax output), trained by hand-derived backprop — numpy, no
autodiff library. Deliberately tiny: SISA trains dozens of these (one per shard, incrementally per slice), so
each one has to be cheap enough that "retrain one slice" is something you can actually feel the speed of.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MLP:
    W1: np.ndarray    # d x h
    b1: np.ndarray    # h
    W2: np.ndarray    # h x c
    b2: np.ndarray    # c

    def copy(self) -> "MLP":
        return MLP(self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy())


def make_mlp(d: int, h: int, c: int, seed: int) -> MLP:
    rng = np.random.default_rng(seed)
    return MLP(rng.normal(0, (2 / d) ** 0.5, (d, h)), np.zeros(h), rng.normal(0, (2 / h) ** 0.5, (h, c)), np.zeros(c))


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def forward(model: MLP, X: np.ndarray) -> tuple:
    """Returns ``(probs, cache)``; ``cache`` is whatever ``backward`` needs, so a caller that only wants
    predictions can ignore it."""
    z1 = X @ model.W1 + model.b1
    a1 = relu(z1)
    z2 = a1 @ model.W2 + model.b2
    probs = softmax(z2)
    return probs, (X, z1, a1)


def predict_proba(model: MLP, X: np.ndarray) -> np.ndarray:
    return forward(model, X)[0]


def cross_entropy(probs: np.ndarray, y: np.ndarray) -> float:
    n = len(y)
    return float(-np.log(np.clip(probs[np.arange(n), y], 1e-9, 1.0)).mean())


def per_example_loss(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    return -np.log(np.clip(probs[np.arange(len(y)), y], 1e-9, 1.0))


def accuracy(probs: np.ndarray, y: np.ndarray) -> float:
    return float((probs.argmax(axis=1) == y).mean())


def backward(model: MLP, cache: tuple, probs: np.ndarray, y: np.ndarray) -> tuple:
    """d(cross_entropy)/d(every parameter), by the chain rule: softmax-cross-entropy's gradient w.r.t. the
    output logits is ``probs - onehot(y)``; ReLU's local gradient is 1 where its input was positive, 0 where it
    was clipped; everything else is the usual two-layer chain rule."""
    X, z1, a1 = cache
    n = len(y)
    onehot = np.zeros_like(probs)
    onehot[np.arange(n), y] = 1.0
    dz2 = (probs - onehot) / n                    # n x c
    dW2 = a1.T @ dz2                               # h x c
    db2 = dz2.sum(axis=0)                          # c
    da1 = dz2 @ model.W2.T                         # n x h
    dz1 = da1 * (z1 > 0)                           # ReLU backward
    dW1 = X.T @ dz1                                # d x h
    db1 = dz1.sum(axis=0)                          # h
    return dW1, db1, dW2, db2


def sgd_step(model: MLP, grads: tuple, lr: float) -> None:
    dW1, db1, dW2, db2 = grads
    model.W1 -= lr * dW1
    model.b1 -= lr * db1
    model.W2 -= lr * dW2
    model.b2 -= lr * db2


def train_steps(model: MLP, X: np.ndarray, y: np.ndarray, steps: int, lr: float) -> list:
    """Full-batch gradient descent for ``steps`` steps on ``(X, y)``, mutating ``model`` in place. Returns the
    training-loss curve, one entry per step."""
    curve = []
    for _ in range(steps):
        probs, cache = forward(model, X)
        curve.append(cross_entropy(probs, y))
        grads = backward(model, cache, probs, y)
        sgd_step(model, grads, lr)
    return curve
