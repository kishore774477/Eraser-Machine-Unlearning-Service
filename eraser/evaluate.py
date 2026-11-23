"""Aggregate many deletions across many seeds, so "memorized points forget louder" and "SISA is faster" are
measured distributions, not one lucky demo."""
from __future__ import annotations

import statistics as st
from dataclasses import dataclass

from .audit import audit_and_delete, memorization_risk
from .data import held_out, make_dataset, make_task
from .sisa import build_ensemble

N_SHARDS, N_SLICES = 8, 4


@dataclass
class SweepResult:
    n_seeds: int
    n_per_seed: int
    loss_ratio_memorized: list     # loss_after / loss_before, for points flagged "memorized"
    loss_ratio_typical: list       # same, for points flagged "typical"
    ensemble_acc_before: list
    ensemble_acc_after: list

    def to_dict(self) -> dict:
        def stats(xs):
            if not xs:
                return None
            return {"mean": round(st.mean(xs), 4), "median": round(st.median(xs), 4), "min": round(min(xs), 4), "max": round(max(xs), 4),
                     "doubled_or_more": round(sum(1 for x in xs if x >= 2.0) / len(xs), 4), "n": len(xs)}
        return {"n_seeds": self.n_seeds, "n_per_seed": self.n_per_seed,
                "loss_ratio_memorized": stats(self.loss_ratio_memorized), "loss_ratio_typical": stats(self.loss_ratio_typical),
                "ensemble_acc_before": stats(self.ensemble_acc_before), "ensemble_acc_after": stats(self.ensemble_acc_after)}


def sweep(n_seeds: int = 15, n_per_seed: int = 10, base_seed: int = 500, n_patients: int = 1600, pool: int = 300) -> SweepResult:
    """For each seed, flags the first ``pool`` patients as memorized/typical (no peeking at the hidden task),
    then deletes up to ``n_per_seed // 2`` from each group — a balanced sample, not whichever happens to come
    first by patient ID."""
    from .model import accuracy
    ratio_mem, ratio_typ, acc_before, acc_after = [], [], [], []
    half = max(1, n_per_seed // 2)
    for s in range(n_seeds):
        seed = base_seed + s
        task = make_task(seed)
        ds = make_dataset(task, n_patients, seed=seed)
        test = held_out(task, seed=seed, n=400)
        ens = build_ensemble(ds, N_SHARDS, N_SLICES, seed=seed)
        acc_before.append(accuracy(ens.predict_proba(test.X), test.y))
        candidates = [int(pid) for pid in ds.ids[:pool]]
        flags = {pid: memorization_risk(ens, ds, pid) for pid in candidates}
        memorized = [pid for pid in candidates if flags[pid]][:half]
        typical = [pid for pid in candidates if not flags[pid]][:half]
        for pid in memorized:
            r = audit_and_delete(ens, ds, test, pid)
            ratio_mem.append(r.loss_after / max(r.loss_before, 1e-9))
        for pid in typical:
            r = audit_and_delete(ens, ds, test, pid)
            ratio_typ.append(r.loss_after / max(r.loss_before, 1e-9))
        acc_after.append(accuracy(ens.predict_proba(test.X), test.y))
    return SweepResult(n_seeds, n_per_seed, ratio_mem, ratio_typ, acc_before, acc_after)


def measure_speedup(n_seeds: int = 10, base_seed: int = 700, n_patients: int = 1600) -> dict:
    """The honest version of the plan's "1 of 8 shards retrained in 3s vs 24s" claim: actual wall-clock seconds,
    one deletion per seed, with ``measure_full_rebuild`` turned on (expensive — this is why it is its own,
    smaller sweep rather than folded into the main one)."""
    import statistics as stx
    partials, fulls = [], []
    for s in range(n_seeds):
        seed = base_seed + s
        task = make_task(seed)
        ds = make_dataset(task, n_patients, seed=seed)
        test = held_out(task, seed=seed, n=400)
        ens = build_ensemble(ds, N_SHARDS, N_SLICES, seed=seed)
        pid = int(ds.ids[len(ds.ids) // 2])
        r = audit_and_delete(ens, ds, test, pid, measure_full_rebuild=True)
        partials.append(r.receipt.partial_seconds)
        fulls.append(r.receipt.full_rebuild_seconds)
    return {"n_seeds": n_seeds, "partial_seconds": {"mean": round(stx.mean(partials), 5), "max": round(max(partials), 5)},
            "full_rebuild_seconds": {"mean": round(stx.mean(fulls), 5), "max": round(max(fulls), 5)},
            "speedup": round(stx.mean(fulls) / stx.mean(partials), 2)}
