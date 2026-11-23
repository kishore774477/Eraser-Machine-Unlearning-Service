"""Auditing a deletion: not just "is the point gone from the training set" (trivially true the moment it is
removed) but "does the model's own behaviour on it still look like it was ever trained on it".

``memorization_risk`` flags a patient *without* peeking at the synthetic task's hidden ground truth: it asks
whether the *other* seven shards — which never saw this patient — already disagree with their recorded label.
Disagreement means this patient's own shard had nothing to generalise from and had to memorise the label
specifically; agreement means the label follows the pattern everyone else already learned, and removing it
barely changes anything. This is exactly what shows up in practice: deleting a "disagrees" patient swings the
model's loss on them by orders of magnitude; deleting an "agrees" one barely moves it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import row_of, subset
from .mia import train_attack
from .model import per_example_loss, predict_proba
from .sisa import Ensemble
from .unlearn import DeletionReceipt, delete_patient


def leave_shard_out_proba(ens: Ensemble, shard_idx: int, X: np.ndarray) -> np.ndarray:
    others = [s for i, s in enumerate(ens.shards) if i != shard_idx]
    return np.mean([predict_proba(s.model, X) for s in others], axis=0)


def memorization_risk(ens: Ensemble, ds, patient_id: int) -> bool:
    """True if the patient's label disagrees with what the rest of the federation (never having seen them)
    would have predicted — a point this patient's own shard had to memorise rather than generalise to."""
    shard_idx, _ = ens.locate(patient_id)
    row = row_of(ds, patient_id)
    pred = leave_shard_out_proba(ens, shard_idx, ds.X[row:row + 1]).argmax(axis=1)[0]
    return int(pred) != int(ds.y[row])


def member_dataset(ens: Ensemble, ds, shard_idx: int):
    shard = ens.shards[shard_idx]
    member_ids = set(int(i) for s in shard.slices for i in s.ids)
    rows = [i for i, pid in enumerate(ds.ids) if int(pid) in member_ids]
    return subset(ds, rows)


@dataclass
class AuditResult:
    receipt: DeletionReceipt
    memorized: bool
    loss_before: float
    loss_after: float
    attack_score_before: float
    attack_score_after: float

    def to_dict(self) -> dict:
        d = self.receipt.to_dict()
        d.update(memorized=self.memorized, loss_before=round(self.loss_before, 4), loss_after=round(self.loss_after, 4),
                  attack_score_before=round(self.attack_score_before, 4), attack_score_after=round(self.attack_score_after, 4))
        return d


def audit_and_delete(ens: Ensemble, ds, held_out_ds, patient_id: int, measure_full_rebuild: bool = False) -> AuditResult:
    """Scores the attack on this patient, deletes them for real, then scores the (now-retrained) model on the
    same point with the same attack — the whole point of running the attack twice."""
    shard_idx, _ = ens.locate(patient_id)
    row = row_of(ds, patient_id)
    memorized = memorization_risk(ens, ds, patient_id)
    member_ds = member_dataset(ens, ds, shard_idx)
    attack = train_attack(ens.shards[shard_idx].model, member_ds.X, member_ds.y, held_out_ds.X, held_out_ds.y, seed=0)
    loss_before = float(per_example_loss(predict_proba(ens.shards[shard_idx].model, ds.X[row:row + 1]), ds.y[row:row + 1])[0])
    score_before = float(attack.score(ens.shards[shard_idx].model, ds.X[row:row + 1], ds.y[row:row + 1])[0])
    receipt = delete_patient(ens, ds, patient_id, measure_full_rebuild)
    model_after = ens.shards[shard_idx].model
    loss_after = float(per_example_loss(predict_proba(model_after, ds.X[row:row + 1]), ds.y[row:row + 1])[0])
    score_after = float(attack.score(model_after, ds.X[row:row + 1], ds.y[row:row + 1])[0])
    return AuditResult(receipt, memorized, loss_before, loss_after, score_before, score_after)
