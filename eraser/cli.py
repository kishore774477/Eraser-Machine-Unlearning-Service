"""Command line: python -m eraser <command>"""
from __future__ import annotations

import argparse
import sys

from .audit import audit_and_delete, memorization_risk
from .data import held_out, make_dataset, make_task
from .evaluate import measure_speedup, sweep
from .model import accuracy
from .sisa import build_ensemble


def main(argv=None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(prog="eraser", description="Machine unlearning: SISA sharded training, a deletion API, and a membership-inference audit")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--slices", type=int, default=4)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build", help="train the SISA ensemble and report accuracy")
    p = sub.add_parser("delete", help="delete one patient, with a full membership-inference audit")
    p.add_argument("patient_id", type=int)
    p.add_argument("--full-rebuild", action="store_true", help="also measure a full ensemble rebuild, for the time comparison (slow)")
    p = sub.add_parser("sweep", help="many deletions, many seeds: memorized vs typical points")
    p.add_argument("--seeds", type=int, default=15)
    p.add_argument("--per-seed", type=int, default=10)
    p = sub.add_parser("speedup", help="measure partial vs full-rebuild wall-clock time, many seeds")
    p.add_argument("--seeds", type=int, default=10)
    a = ap.parse_args(argv)

    task = make_task(a.seed)
    ds = make_dataset(task, 1600, seed=a.seed)
    test = held_out(task, seed=a.seed, n=400)

    if a.cmd == "build":
        ens = build_ensemble(ds, a.shards, a.slices, seed=a.seed)
        print(f"{a.shards} shards x {a.slices} slices, {len(ds.ids)} patients, seed {a.seed}")
        print(f"ensemble accuracy on held-out patients: {accuracy(ens.predict_proba(test.X), test.y):.1%}")
        print(f"total training time: {ens.total_cost():.3f}s")
    elif a.cmd == "delete":
        ens = build_ensemble(ds, a.shards, a.slices, seed=a.seed)
        try:
            flagged = memorization_risk(ens, ds, a.patient_id)
            r = audit_and_delete(ens, ds, test, a.patient_id, measure_full_rebuild=a.full_rebuild)
        except KeyError as e:
            sys.exit(str(e))
        d = r.to_dict()
        print(f"patient {a.patient_id}: shard {d['shard']}, slice {d['slice']}{'  (flagged: likely memorized)' if flagged else '  (flagged: fits the general pattern)'}")
        print(f"loss on this point:  {d['loss_before']:.4f} -> {d['loss_after']:.4f}")
        print(f"attack's member-probability:  {d['attack_score_before']:.3f} -> {d['attack_score_after']:.3f}")
        print(f"retrained in {d['partial_seconds']:.4f}s" + (f" (a full rebuild took {d['full_rebuild_seconds']:.4f}s, {d['speedup']:.1f}x slower)" if d["speedup"] else ""))
    elif a.cmd == "sweep":
        s = sweep(a.seeds, a.per_seed).to_dict()
        print(f"{s['n_seeds']} seeds x {s['n_per_seed']} deletions each (balanced memorized/typical)\n")
        for k in ("loss_ratio_memorized", "loss_ratio_typical"):
            v = s[k]
            print(f"{k}: median x{v['median']:.2f}, {v['doubled_or_more']:.0%} at least doubled (n={v['n']})")
        print(f"ensemble accuracy: {s['ensemble_acc_before']['mean']:.1%} -> {s['ensemble_acc_after']['mean']:.1%}")
    elif a.cmd == "speedup":
        print(measure_speedup(a.seeds))


if __name__ == "__main__":
    main()
