"""Build an ensemble, delete one patient, and print the before/after audit.

    python examples/delete_a_patient.py            # a patient the system flags as likely memorized
    python examples/delete_a_patient.py --typical   # a patient that fits the general pattern instead
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eraser import audit_and_delete, build_ensemble, held_out, make_dataset, make_task, memorization_risk  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--typical", action="store_true", help="pick a patient that fits the general pattern, instead of one flagged as memorized")
a = ap.parse_args()

task = make_task(a.seed)
ds = make_dataset(task, 1600, seed=a.seed)
test = held_out(task, seed=a.seed, n=400)
ens = build_ensemble(ds, n_shards=8, n_slices=4, seed=a.seed)

want_memorized = not a.typical
pid = next(int(pid) for pid in ds.ids[:300] if memorization_risk(ens, ds, int(pid)) == want_memorized)
print(f"patient {pid}: {'likely memorized' if want_memorized else 'fits the general pattern'} (per the other seven shards)\n")

r = audit_and_delete(ens, ds, test, pid, measure_full_rebuild=True)
ratio = f"x{r.loss_after / max(r.loss_before, 1e-9):.1f}" if r.loss_before > 1e-6 else "both already ~0, not a meaningful ratio"
print(f"loss on this point:          {r.loss_before:.4f} -> {r.loss_after:.4f}  ({ratio})")
print(f"attack's member-probability: {r.attack_score_before:.3f} -> {r.attack_score_after:.3f}")
print(f"retrained shard {r.receipt.shard} from slice {r.receipt.slice} in {r.receipt.partial_seconds:.4f}s")
print(f"a full ensemble rebuild took {r.receipt.full_rebuild_seconds:.4f}s ({r.receipt.speedup:.1f}x slower)")
