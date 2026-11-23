"""A synthetic hospital-readmission dataset: one row per patient, a numeric ID a deletion request names, and
features drawn from a fixed (unknown to any trainer) rule plus noise — a genuine pattern to learn, not label
noise, and no member of it is special except whichever one a deletion request happens to name.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

D, CLASSES = 10, 2        # ten features; "readmitted within 30 days" yes/no
FEATURE_NAMES = ("age", "prior_visits", "chronic_conditions", "length_of_stay", "medication_count",
                  "lab_abnormal_count", "er_visits_1y", "bmi", "systolic_bp", "glucose")
LABEL_NOISE = 0.5


@dataclass
class Task:
    w: np.ndarray
    b: float


def make_task(seed: int) -> Task:
    """The fixed rule every patient's label comes from — one task per ``seed``, shared by the training pool and
    the held-out set, so "accuracy on held-out data" means something (a different rule per split would make
    held-out accuracy meaningless, not just hard)."""
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 1.0, D)
    w[1] *= 1.8   # prior_visits matters more
    w[2] *= 1.5   # chronic_conditions matters more
    return Task(w, float(rng.normal(0, 0.3)))


@dataclass
class Dataset:
    ids: np.ndarray        # patient IDs, 1000, 1001, ...
    X: np.ndarray          # n x D
    y: np.ndarray          # n, in {0, 1}
    id_index: dict         # patient id -> row position, for O(1) lookup


def make_dataset(task: Task, n: int, seed: int, start_id: int = 1000) -> Dataset:
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, D))
    logit = X @ task.w + task.b + rng.normal(0, LABEL_NOISE, n)     # some outcomes are genuinely unpredictable
    y = (logit > 0).astype(int)
    ids = np.arange(start_id, start_id + n)
    return Dataset(ids, X, y, {int(i): k for k, i in enumerate(ids)})


def held_out(task: Task, seed: int, n: int = 400) -> Dataset:
    """A clean set nobody ever trains on — the yardstick for ensemble accuracy, and the "known non-members" the
    membership-inference attack calibrates against. Same task, a disjoint ID range, a different sampling seed."""
    return make_dataset(task, n, seed=seed + 777, start_id=900_000)


def row_of(ds: Dataset, patient_id: int) -> int:
    if patient_id not in ds.id_index:
        raise KeyError(f"no such patient: {patient_id}")
    return ds.id_index[patient_id]


def subset(ds: Dataset, rows) -> Dataset:
    rows = np.asarray(rows)
    return Dataset(ds.ids[rows], ds.X[rows], ds.y[rows], {int(i): k for k, i in enumerate(ds.ids[rows])})
