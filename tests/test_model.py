import numpy as np
import pytest

from eraser.model import (MLP, accuracy, backward, cross_entropy, forward, make_mlp, per_example_loss,
                           predict_proba, relu, sgd_step, softmax, train_steps)


def test_softmax_rows_sum_to_one_and_are_shift_invariant():
    z = np.array([[1.0, 2.0, 3.0], [100.0, 100.0, 101.0]])
    p = softmax(z)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert np.allclose(softmax(z - 50), p)


def test_relu_zeroes_negatives_only():
    z = np.array([-2.0, -0.1, 0.0, 0.1, 5.0])
    assert np.array_equal(relu(z), [0, 0, 0, 0.1, 5.0])


def test_gradient_matches_numerical_gradient():
    rng = np.random.default_rng(3)
    m = make_mlp(5, 4, 3, seed=2)
    X, y = rng.normal(size=(15, 5)), rng.integers(0, 3, 15)
    probs, cache = forward(m, X)
    grads = backward(m, cache, probs, y)

    def loss(model):
        p, _ = forward(model, X)
        return cross_entropy(p, y)

    eps = 1e-5
    for name, g in zip(("W1", "b1", "W2", "b2"), grads):
        arr = getattr(m, name)
        num = np.zeros_like(arr)
        for idx in np.ndindex(arr.shape):
            m2 = m.copy(); getattr(m2, name)[idx] += eps
            m3 = m.copy(); getattr(m3, name)[idx] -= eps
            num[idx] = (loss(m2) - loss(m3)) / (2 * eps)
        assert np.abs(num - g).max() < 1e-6, name


def test_train_steps_reduces_loss_and_overfits_a_tiny_set():
    rng = np.random.default_rng(1)
    m = make_mlp(8, 12, 2, seed=0)
    X, y = rng.normal(size=(30, 8)), rng.integers(0, 2, 30)
    curve = train_steps(m, X, y, steps=150, lr=0.4)
    assert curve[-1] < curve[0] * 0.5
    assert accuracy(predict_proba(m, X), y) > 0.9


def test_per_example_loss_matches_cross_entropy_mean():
    probs = np.array([[0.9, 0.1], [0.3, 0.7], [0.5, 0.5]])
    y = np.array([0, 1, 1])
    per = per_example_loss(probs, y)
    assert per_example_loss(probs, y).mean() == pytest.approx(cross_entropy(probs, y))
    assert per[2] == pytest.approx(-np.log(0.5))


def test_accuracy_on_a_known_case():
    probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.3, 0.7]])
    y = np.array([0, 1, 0])
    assert accuracy(probs, y) == pytest.approx(2 / 3)


def test_mlp_copy_is_independent():
    m = make_mlp(4, 3, 2, seed=0)
    c = m.copy()
    c.W1[0, 0] += 100
    assert m.W1[0, 0] != c.W1[0, 0]
