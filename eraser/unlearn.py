"""The deletion API: find the one shard and slice a patient lives in, remove them, and retrain only what their
removal could possibly have touched — never the other seven shards, never the slices before theirs.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .data import subset
from .sisa import Ensemble, build_ensemble


@dataclass
class DeletionReceipt:
    patient_id: int
    shard: int
    slice: int
    partial_seconds: float
    full_rebuild_seconds: float | None   # measured only when asked for; expensive, so not on every deletion

    @property
    def speedup(self) -> float | None:
        return None if self.full_rebuild_seconds is None else self.full_rebuild_seconds / max(self.partial_seconds, 1e-9)

    def to_dict(self) -> dict:
        return {"patient_id": self.patient_id, "shard": self.shard, "slice": self.slice, "partial_seconds": round(self.partial_seconds, 5),
                "full_rebuild_seconds": round(self.full_rebuild_seconds, 5) if self.full_rebuild_seconds is not None else None,
                "speedup": round(self.speedup, 2) if self.speedup is not None else None}


def delete_patient(ens: Ensemble, ds, patient_id: int, measure_full_rebuild: bool = False) -> DeletionReceipt:
    """Removes ``patient_id`` from the ensemble's training data for good and retrains only the one shard it
    lived in, from the checkpoint before its slice. ``ds`` is mutated nowhere — only the shard's own slice
    copies are; the caller's original dataset is untouched, so a repeated audit elsewhere still sees the patient
    unless it is told to."""
    from .sisa import retrain_from_slice
    shard_idx, slice_idx = ens.locate(patient_id)
    shard = ens.shards[shard_idx]
    sl = shard.slices[slice_idx]
    if patient_id not in sl.id_index:
        raise KeyError(f"no such patient: {patient_id}")
    keep = sl.ids != patient_id
    new_slices = [subset(sl, np.where(keep)[0])] + shard.slices[slice_idx + 1:]
    partial = retrain_from_slice(shard, slice_idx, new_slices, ens.hidden, ens.steps_per_slice, ens.lr, ens.task_seed)
    full_seconds = None
    if measure_full_rebuild:
        rows = np.array([i for i in range(len(ds.ids)) if int(ds.ids[i]) != patient_id])
        t0 = time.perf_counter()
        build_ensemble(subset(ds, rows), len(ens.shards), max(len(s.slices) for s in ens.shards), ens.task_seed, ens.hidden, ens.steps_per_slice, ens.lr)
        full_seconds = time.perf_counter() - t0
    return DeletionReceipt(patient_id, shard_idx, slice_idx, partial, full_seconds)
