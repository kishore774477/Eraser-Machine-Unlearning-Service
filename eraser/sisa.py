"""SISA: Sharded, Isolated, Sliced, Aggregated training (Bourtoule et al.'s machine-unlearning construction).

Every patient belongs to exactly one shard — shards share no data, so deleting a point can never require
touching any shard but its own. Within a shard, training data is split into an ordered sequence of *slices* and
trained incrementally, checkpointing after each one: slice 0, checkpoint; add slice 1, keep training, checkpoint;
and so on. Deleting a point in slice *i* only invalidates the checkpoints from slice *i* onward — slices
`0..i-1` already reflect a world without that point's influence on anything after them, so retraining can resume
from the checkpoint *before* slice *i* instead of starting that shard over.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .data import Dataset, subset
from .model import MLP, make_mlp, predict_proba, train_steps

HIDDEN, STEPS_PER_SLICE, LR = 24, 120, 0.5     # deliberately enough capacity and steps to memorise a 200-row shard
# (test accuracy stays reasonable; see the README for the measured train/test gap this creates, and why the
# membership-inference audit depends on it: an audit against a model with no such gap has nothing to detect)


@dataclass
class Shard:
    index: int
    slices: list                 # list[Dataset], one per slice, in their fixed training order
    checkpoints: list = field(default_factory=list)    # checkpoints[i] = model state *after* training slice i
    cost: float = 0.0            # total wall-clock seconds spent training this shard, ever

    @property
    def model(self) -> MLP:
        return self.checkpoints[-1]

    def point_slice(self, patient_id: int) -> int | None:
        for i, s in enumerate(self.slices):
            if patient_id in s.id_index:
                return i
        return None


def assign_shards(ds: Dataset, n_shards: int, n_slices: int, seed: int) -> list:
    """A fixed, random (but reproducible) assignment of every row to one shard and, within that shard, to one
    slice — decided once, at ingestion time, exactly as SISA requires: the split cannot depend on anything a
    later deletion would change."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ds.ids))
    shard_of = np.array_split(order, n_shards)
    shards_rows = []
    for rows in shard_of:
        slice_of = np.array_split(rows, n_slices)
        shards_rows.append([subset(ds, r) for r in slice_of if len(r)])
    return shards_rows


def train_shard(index: int, slices: list, hidden: int = HIDDEN, steps_per_slice: int = STEPS_PER_SLICE, lr: float = LR,
                 seed: int = 0) -> Shard:
    shard = Shard(index, slices)
    model = make_mlp(slices[0].X.shape[1], hidden, 2, seed=seed + index)
    seen_X, seen_y = None, None
    for s in slices:
        seen_X = s.X if seen_X is None else np.concatenate([seen_X, s.X])
        seen_y = s.y if seen_y is None else np.concatenate([seen_y, s.y])
        t0 = time.perf_counter()
        train_steps(model, seen_X, seen_y, steps_per_slice, lr)
        shard.cost += time.perf_counter() - t0
        shard.checkpoints.append(model.copy())
    return shard


def retrain_from_slice(shard: Shard, start_slice: int, new_slices: list, hidden: int = HIDDEN,
                        steps_per_slice: int = STEPS_PER_SLICE, lr: float = LR, seed: int = 0) -> float:
    """Resume training a shard from the checkpoint *before* ``start_slice``, replaying ``new_slices`` (the
    updated slices from ``start_slice`` on — one of them has had a point removed) forward. Mutates ``shard`` in
    place and returns the wall-clock seconds spent, which is the number this project's "time saved" claim is
    built from — never a guess."""
    model = shard.checkpoints[start_slice - 1].copy() if start_slice > 0 else make_mlp(new_slices[0].X.shape[1], hidden, 2, seed=seed + shard.index)
    seen_X = np.concatenate([s.X for s in shard.slices[:start_slice]]) if start_slice > 0 else None
    seen_y = np.concatenate([s.y for s in shard.slices[:start_slice]]) if start_slice > 0 else None
    t0 = time.perf_counter()
    for i, s in enumerate(new_slices):
        seen_X = s.X if seen_X is None else np.concatenate([seen_X, s.X])
        seen_y = s.y if seen_y is None else np.concatenate([seen_y, s.y])
        train_steps(model, seen_X, seen_y, steps_per_slice, lr)
        shard.checkpoints[start_slice + i] = model.copy()
    cost = time.perf_counter() - t0
    shard.slices[start_slice:] = new_slices
    shard.checkpoints = shard.checkpoints[:start_slice + len(new_slices)]
    shard.cost += cost
    return cost


@dataclass
class Ensemble:
    shards: list
    task_seed: int
    hidden: int = HIDDEN
    steps_per_slice: int = STEPS_PER_SLICE
    lr: float = LR

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """SISA's aggregation step: average each shard's own prediction — a plain, unweighted vote, since no
        shard is supposed to be more trustworthy than another."""
        return np.mean([predict_proba(s.model, X) for s in self.shards], axis=0)

    def locate(self, patient_id: int) -> tuple:
        for shard in self.shards:
            sl = shard.point_slice(patient_id)
            if sl is not None:
                return shard.index, sl
        raise KeyError(f"no such patient: {patient_id}")

    def total_cost(self) -> float:
        return sum(s.cost for s in self.shards)


def build_ensemble(ds: Dataset, n_shards: int, n_slices: int, seed: int, hidden: int = HIDDEN,
                    steps_per_slice: int = STEPS_PER_SLICE, lr: float = LR) -> Ensemble:
    shards_rows = assign_shards(ds, n_shards, n_slices, seed)
    shards = [train_shard(i, slices, hidden, steps_per_slice, lr, seed) for i, slices in enumerate(shards_rows)]
    return Ensemble(shards, seed, hidden, steps_per_slice, lr)
