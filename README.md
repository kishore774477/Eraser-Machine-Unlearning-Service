# Eraser: delete a patient from a trained model without a full retrain, then prove it

A deletion request ("forget patient #1004") is easy to satisfy badly: drop the row from a database and call it
done. The model trained on that row still carries whatever it learned from it — a dropped row proves nothing
about the model. **Eraser does the actual unlearning** (SISA: sharded, isolated, sliced training, so only one
shard ever needs retraining) and then **proves it with a membership-inference audit** — the same kind of attack
a privacy researcher would use to check *you*, run against your own model, before and after.

```
 1,600 patients, sharded 8 ways at ingestion (shards never share data), each shard sliced into 4 ordered pieces
        │  train incrementally: slice 0 -> checkpoint -> + slice 1 -> checkpoint -> ... -> + slice 3
        ▼
 ensemble = plain average of all 8 shards' own predictions        (92-95% accuracy on held-out patients)
        ▼
 delete patient #1004: find their shard + slice -> remove them -> resume from the checkpoint BEFORE that slice
        │  (only that shard changes; only the slices from theirs onward are redone)
        ▼
 audit: a membership-inference attack scores "was this point a training member?" — once before, once after
```

## What makes it different

| Idea | What it does |
|---|---|
| **Deletion is a resume, not a rebuild** | Retraining replays only the affected shard, only from the checkpoint before the affected slice. Measured on this project's own ensemble: a partial retrain averages **0.020s**; rebuilding all 8 shards from scratch averages **0.224s** — **11x** slower. |
| **A deletion is only proved by an audit, never by a lookup** | The point is gone from the training set the moment it is removed — trivially, that proves nothing about whether the *model* still carries its influence. The audit scores the retrained shard's own behaviour on the deleted point, which is the only thing that can actually show a change. |
| **Not every point forgets equally loud, and the system can tell you which** | A patient is flagged "likely memorized" using no privileged knowledge — only whether the *other seven shards* (which never saw them) already disagree with their recorded label. That flag predicts the deletion's visible effect with real precision (72%) and recall (86%) against the hidden ground truth, measured without ever consulting it to build the flag. |
| **The audit's own honesty is reported, not assumed** | Its population-wide AUC (can it tell members from non-members *in general*?) is measured and reported — and it is weak (~0.55), because a model that generalises this well leaves little membership signal on a typical point. The dramatic effect lives in the minority of memorized points, and the README says so instead of implying the attack is universally powerful. |

## See it work (no API key needed — it is a self-contained simulation)

```
python -m eraser delete 1004 --full-rebuild

patient 1004: shard 2, slice 1  (flagged: likely memorized)
loss on this point:  0.0282 -> 0.1399
attack's member-probability:  0.318 -> 0.315
retrained in 0.0215s (a full rebuild took 0.2110s, 9.8x slower)
```

### Held out: 25 seeds, 10 deletions each, balanced memorized/typical

| | Memorized points | Typical points |
|---|---:|---:|
| **Median loss, after ÷ before** | **x26.7** | x0.99 |
| Fraction that at least doubled | **91.2%** | 20.0% |
| n | 125 | 125 |

Ensemble accuracy on held-out patients: **94.9%** before these deletions, **94.8%** after — removing one point at
a time barely moves it, as it should not.

### What this project found, including the parts that were less clean than hoped

* **A three-feature attack (loss, top confidence, prediction entropy) turned out to be numerically unstable.**
  In a two-class model, entropy is a deterministic function of the top-class confidence — the two features are
  collinear — and the fitted logistic regression sometimes landed on the *wrong-signed* coefficient for entropy,
  occasionally moving a deleted point's score the wrong direction. Dropping to the single standard feature
  (`log(loss)`, the classic Yeom et al. 2018 baseline) fixed it and is simpler besides.
* **The population-wide membership-inference attack is weak on this ensemble — about 0.55 AUC, barely above the
  0.5 a coin flip gets.** That is not a bug: this ensemble generalises to held-out patients at 92-95% accuracy,
  and a model that generalises well has little gap between how it treats members and non-members on a *typical*
  point. Early testing with a label-flip "poisoning" task (same lesson, different project this week) found the
  same thing from the other direction — strong effects need either genuine memorization or a genuinely
  surprising point, not an average one.
* **The dramatic effect is concentrated in points the model had no choice but to memorize** — ones where the
  rest of the federation (who never saw them) already disagrees with the recorded label. There is no generic
  pattern to lean on for those, so the model's only option was to memorize the specific exception, and deleting
  it removes exactly that memorized exception. This is also why the "memorization risk" flag is a feature here,
  not an afterthought: which points are actually worth auditing after a deletion is itself useful information.

## Run it

```powershell
start.bat                        # creates .venv, installs, opens http://127.0.0.1:8190
# or
python -m venv .venv ; .venv\Scripts\pip install -r requirements.txt ; .venv\Scripts\python run.py
```

Dashboard tabs: **Shards** (every shard's slices and training cost), **Delete a patient** (browse patients with
a "likely memorized" / "typical" flag on each, delete one, and see the full before/after audit — loss, attack
score, timing, with an optional full-rebuild comparison), **Deletion log** (every deletion this session),
**Evidence** (the many-seed comparison, live), **How it works**.

```powershell
python -m eraser build                            # train the ensemble, report held-out accuracy
python -m eraser delete 1004 --full-rebuild        # delete one patient, with the full audit and timing
python -m eraser sweep --seeds 25 --per-seed 10    # the memorized-vs-typical comparison, many seeds
python -m eraser speedup --seeds 15                # partial vs full-rebuild wall-clock time, many seeds
python examples\delete_a_patient.py                 # the library in 25 lines
python examples\delete_a_patient.py --typical       # the same, for a patient that is NOT flagged
```

```python
from eraser import make_task, make_dataset, held_out, build_ensemble, audit_and_delete
task = make_task(1)
ds = make_dataset(task, 1600, seed=1)
ens = build_ensemble(ds, n_shards=8, n_slices=4, seed=1)
r = audit_and_delete(ens, ds, held_out(task, seed=1), patient_id=1004, measure_full_rebuild=True)
r.loss_before, r.loss_after, r.receipt.speedup
```

| HTTP | |
|---|---|
| `GET /api/meta` `/api/shards` | the ensemble's accuracy and shape; every shard's slices and training cost |
| `GET /api/patients` · `/api/patient/{id}` | browse patients with their shard/slice/memorization flag; one patient's current loss |
| `POST /api/delete` | `{patient_id, full_rebuild}` → delete, retrain, and audit; returns the before/after numbers |
| `GET /api/log` | every deletion made this session |
| `GET /api/sweep` · `POST /api/sweep` | the cached many-seed comparison (computed in the background on startup); force a fresh one with `?n_seeds=&per_seed=` |

Configure with `ERASER_PORT`, `ERASER_SEED` (see `.env.example`). There is no API key anywhere in this project.

## Tests

```powershell
.venv\Scripts\python -m pytest -q        # 47 tests, ~30 s (most are instant; the sweeps and background-precompute test run real ensembles)
```

They check: the MLP's hand-derived backprop against a numerical gradient (max error ~1e-11) for all four
parameter tensors; training reduces loss and overfits a tiny set; the training task and the held-out set share
one task rule (a regression against an earlier version where they silently didn't, making held-out accuracy
meaningless rather than merely hard); shard assignment is a disjoint, complete, deterministic partition; a
partial retrain changes only the affected shard and nothing before the affected slice; retraining from the last
slice costs less than from the first; the membership-inference AUC formula against hand-built cases; an overfit
model is more attackable than a fresh one; deleting a patient really removes them (`locate` then raises) without
mutating the original dataset; the headline (memorized points forget far louder than typical ones) on seeds
distinct from the ones used while building this; the API and CLI, including the background precompute actually
finishing.

## Limitations

* 1,600 synthetic patients and 8 shards are sized so every demo in this dashboard runs in well under a second; the absolute speedup number would look different, not necessarily smaller, on a production-scale model and dataset — the *mechanism* (resume from a checkpoint instead of restarting) is what transfers, not this exact multiplier.
* The membership-inference attack here is the simplest standard baseline (loss-based, Yeom et al. 2018), not a shadow-model attack — a stronger attacker would likely find more signal than the ~0.55 AUC measured here, which should be read as a floor on leakage, not a ceiling.
* "Likely memorized" is a heuristic (disagreement with a leave-one-shard-out ensemble vote), not a certificate; it has real but imperfect precision and recall against the hidden ground truth in this synthetic dataset, and nothing guarantees it generalizes to how a real dataset's hard cases would look.
* Only single-point deletions are implemented; deleting many points from the same shard at once, or deletion requests that arrive faster than shards can retrain, are not addressed.
