"""Eraser: delete one patient from a trained model without a full retrain (SISA sharded + sliced training), then
prove the deletion actually changed the model, not just a lookup table, with a membership-inference audit.

    from eraser import make_task, make_dataset, held_out, build_ensemble, audit_and_delete
    task = make_task(1)
    ds = make_dataset(task, 1600, seed=1)
    ens = build_ensemble(ds, n_shards=8, n_slices=4, seed=1)
    r = audit_and_delete(ens, ds, held_out(task, seed=1), patient_id=int(ds.ids[5]))
    r.loss_before, r.loss_after           # how surprised the model is by this point, before and after
"""
from __future__ import annotations

from .audit import AuditResult, audit_and_delete, memorization_risk
from .data import Dataset, Task, held_out, make_dataset, make_task
from .sisa import Ensemble, build_ensemble
from .unlearn import DeletionReceipt, delete_patient

__version__ = "0.1.0"
__all__ = ["AuditResult", "Dataset", "DeletionReceipt", "Ensemble", "Task", "audit_and_delete", "build_ensemble",
           "delete_patient", "held_out", "make_dataset", "make_task", "memorization_risk", "__version__"]
