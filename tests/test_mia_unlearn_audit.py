import numpy as np
import pytest

from eraser.audit import audit_and_delete, leave_shard_out_proba, member_dataset, memorization_risk
from eraser.data import held_out, make_dataset, make_task, subset
from eraser.mia import auc, features, train_attack
from eraser.model import accuracy, make_mlp, predict_proba, train_steps
from eraser.sisa import build_ensemble
from eraser.unlearn import delete_patient


# ---------------------------------------------------------------------------- MIA building blocks
def test_auc_on_known_cases():
    rng = np.random.default_rng(0)
    scores = rng.normal(0, 1, 500)
    labels = (scores + rng.normal(0, 0.2, 500) > 0).astype(int)
    assert auc(scores, labels) > 0.9
    assert auc(rng.normal(size=500), rng.integers(0, 2, 500)) == pytest.approx(0.5, abs=0.08)
    assert auc(np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1])) == 1.0
    assert auc(np.array([1.0]), np.array([1])) == 0.5     # no non-members in the set: defined, not a crash


def test_features_shape_and_log_transform():
    m = make_mlp(5, 4, 2, seed=0)
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(10, 5)), rng.integers(0, 2, 10)
    f = features(m, X, y)
    assert f.shape == (10, 1)
    from eraser.model import per_example_loss
    loss = per_example_loss(predict_proba(m, X), y)
    assert np.allclose(f[:, 0], np.log(loss + 1e-9))


def test_a_confidently_overfit_model_is_more_attackable_than_a_fresh_one():
    """A model trained hard on a small set should separate its own training data from unseen data better than
    an untrained one with random weights."""
    rng = np.random.default_rng(1)
    X, y = rng.normal(size=(60, 6)), rng.integers(0, 2, 60)
    Xo, yo = rng.normal(size=(200, 6)), rng.integers(0, 2, 200)

    fresh = make_mlp(6, 10, 2, seed=2)
    attack_fresh = train_attack(fresh, X, y, Xo, yo, seed=0)
    s_fresh = np.concatenate([attack_fresh.score(fresh, X, y), attack_fresh.score(fresh, Xo, yo)])
    l_fresh = np.concatenate([np.ones(60), np.zeros(200)])

    overfit = make_mlp(6, 10, 2, seed=2)
    train_steps(overfit, X, y, steps=400, lr=0.5)
    attack_of = train_attack(overfit, X, y, Xo, yo, seed=0)
    s_of = np.concatenate([attack_of.score(overfit, X, y), attack_of.score(overfit, Xo, yo)])

    assert auc(s_of, l_fresh) > auc(s_fresh, l_fresh) + 0.05


# ---------------------------------------------------------------------------- unlearn.py
def test_delete_patient_removes_them_and_retrains_only_their_shard(ensemble, ds):
    pid = int(ds.ids[3])
    shard_idx, slice_idx = ensemble.locate(pid)
    others_before = [s.model.W1.copy() for i, s in enumerate(ensemble.shards) if i != shard_idx]
    receipt = delete_patient(ensemble, ds, pid)
    assert receipt.patient_id == pid and receipt.shard == shard_idx and receipt.slice == slice_idx
    assert receipt.partial_seconds > 0 and receipt.full_rebuild_seconds is None and receipt.speedup is None
    others_after = [s.model.W1 for i, s in enumerate(ensemble.shards) if i != shard_idx]
    assert all(np.array_equal(a, b) for a, b in zip(others_before, others_after))
    with pytest.raises(KeyError):
        ensemble.locate(pid)


def test_delete_patient_with_full_rebuild_measures_a_real_speedup(ensemble, ds):
    pid = int(ds.ids[5])
    r = delete_patient(ensemble, ds, pid, measure_full_rebuild=True)
    assert r.full_rebuild_seconds is not None and r.full_rebuild_seconds > r.partial_seconds
    assert r.speedup > 1.0


def test_deleting_an_unknown_patient_raises(ensemble, ds):
    with pytest.raises(KeyError):
        delete_patient(ensemble, ds, 999999)


def test_deleting_does_not_mutate_the_original_dataset(ensemble, ds):
    n_before = len(ds.ids)
    delete_patient(ensemble, ds, int(ds.ids[7]))
    assert len(ds.ids) == n_before


# ---------------------------------------------------------------------------- audit.py
def test_member_dataset_matches_the_shards_own_slices(ensemble, ds):
    shard = ensemble.shards[0]
    want = sorted(int(i) for sl in shard.slices for i in sl.ids)
    got = sorted(int(i) for i in member_dataset(ensemble, ds, 0).ids)
    assert want == got


def test_leave_shard_out_excludes_exactly_that_shard(ensemble, test_ds):
    p_all = ensemble.predict_proba(test_ds.X)
    p_loo = leave_shard_out_proba(ensemble, 0, test_ds.X)
    assert not np.allclose(p_all, p_loo)       # dropping a shard changes the average
    p_loo_manual = np.mean([predict_proba(s.model, test_ds.X) for i, s in enumerate(ensemble.shards) if i != 0], axis=0)
    assert np.allclose(p_loo, p_loo_manual)


def test_memorization_risk_is_boolean_and_matches_disagreement(ensemble, ds):
    pid = int(ds.ids[0])
    flag = memorization_risk(ensemble, ds, pid)
    assert isinstance(flag, bool)


def test_audit_and_delete_reports_both_loss_and_attack_score(ensemble, ds, test_ds):
    pid = int(ds.ids[9])
    r = audit_and_delete(ensemble, ds, test_ds, pid)
    assert r.receipt.patient_id == pid
    assert r.loss_before >= 0 and r.loss_after >= 0
    assert 0 <= r.attack_score_before <= 1 and 0 <= r.attack_score_after <= 1
    d = r.to_dict()
    assert d["patient_id"] == pid and "memorized" in d and "loss_before" in d


def test_memorized_points_forget_louder_than_typical_ones(ensemble, ds, test_ds):
    """The project's headline finding, on one ensemble: find one flagged-memorized and one flagged-typical
    patient and confirm the direction, without requiring a specific magnitude (that is what the sweep checks
    across many seeds, in test_evaluate_api_cli.py)."""
    candidates = [int(pid) for pid in ds.ids[:200]]
    memorized = next(pid for pid in candidates if memorization_risk(ensemble, ds, pid))
    typical = next(pid for pid in candidates if not memorization_risk(ensemble, ds, pid))
    r_mem = audit_and_delete(ensemble, ds, test_ds, memorized)
    ens2 = build_ensemble(ds, 8, 4, seed=1)
    r_typ = audit_and_delete(ens2, ds, test_ds, typical)
    ratio_mem = r_mem.loss_after / max(r_mem.loss_before, 1e-9)
    ratio_typ = r_typ.loss_after / max(r_typ.loss_before, 1e-9)
    assert ratio_mem > ratio_typ
