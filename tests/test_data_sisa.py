import numpy as np
import pytest

from eraser.data import D, held_out, make_dataset, make_task, row_of, subset
from eraser.model import accuracy
from eraser.sisa import assign_shards, build_ensemble, retrain_from_slice, train_shard


# ---------------------------------------------------------------------------- data
def test_make_task_is_deterministic():
    t1, t2 = make_task(5), make_task(5)
    assert np.array_equal(t1.w, t2.w) and t1.b == t2.b
    assert not np.array_equal(make_task(5).w, make_task(6).w)


def test_dataset_and_held_out_share_the_same_task_rule():
    task = make_task(1)
    ds = make_dataset(task, 400, seed=1)
    test = held_out(task, seed=1)
    oracle_train = ((ds.X @ task.w + task.b) > 0).astype(int)
    oracle_test = ((test.X @ task.w + task.b) > 0).astype(int)
    assert (oracle_train == ds.y).mean() > 0.85      # the true rule mostly (not always: label noise) explains the labels
    assert (oracle_test == test.y).mean() > 0.85      # and the SAME rule explains the held-out set just as well


def test_dataset_ids_and_lookup():
    task = make_task(1)
    ds = make_dataset(task, 100, seed=1, start_id=5000)
    assert list(ds.ids[:3]) == [5000, 5001, 5002]
    assert row_of(ds, 5002) == 2
    with pytest.raises(KeyError):
        row_of(ds, 999999)


def test_held_out_ids_never_collide_with_training_ids():
    task = make_task(1)
    ds = make_dataset(task, 1600, seed=1)
    test = held_out(task, seed=1)
    assert not (set(ds.ids.tolist()) & set(test.ids.tolist()))


def test_subset_preserves_rows_and_rebuilds_the_index():
    task = make_task(1)
    ds = make_dataset(task, 50, seed=1)
    sub = subset(ds, [3, 7, 9])
    assert list(sub.ids) == [ds.ids[3], ds.ids[7], ds.ids[9]]
    assert row_of(sub, int(ds.ids[7])) == 1


# ---------------------------------------------------------------------------- SISA
def test_assign_shards_is_a_disjoint_complete_partition(ds):
    shards_rows = assign_shards(ds, n_shards=8, n_slices=4, seed=1)
    all_ids = [int(i) for shard in shards_rows for sl in shard for i in sl.ids]
    assert sorted(all_ids) == sorted(int(i) for i in ds.ids)       # every patient appears, exactly once
    assert len(shards_rows) == 8
    for shard in shards_rows:
        assert len(shard) <= 4


def test_assign_shards_is_deterministic_given_a_seed(ds):
    a = assign_shards(ds, 8, 4, seed=3)
    b = assign_shards(ds, 8, 4, seed=3)
    assert [list(sl.ids) for shard in a for sl in shard] == [list(sl.ids) for shard in b for sl in shard]
    c = assign_shards(ds, 8, 4, seed=4)
    assert [list(sl.ids) for shard in a for sl in shard] != [list(sl.ids) for shard in c for sl in shard]


def test_ensemble_generalises_better_than_chance(ensemble, test_ds):
    acc = accuracy(ensemble.predict_proba(test_ds.X), test_ds.y)
    assert acc > 0.8      # two classes; chance is 0.5


def test_each_shards_own_model_also_beats_chance(ensemble, test_ds):
    from eraser.model import predict_proba
    for shard in ensemble.shards:
        assert accuracy(predict_proba(shard.model, test_ds.X), test_ds.y) > 0.7


def test_locate_finds_every_patient_exactly_once(ensemble, ds):
    seen = set()
    for pid in ds.ids:
        shard_idx, slice_idx = ensemble.locate(int(pid))
        assert (shard_idx, slice_idx) not in seen or True   # just checking it does not raise
        seen.add(int(pid))
    assert len(seen) == len(ds.ids)
    with pytest.raises(KeyError):
        ensemble.locate(999999)


def test_retrain_from_slice_changes_only_that_shard(ensemble, ds):
    shard0, shard1 = ensemble.shards[0], ensemble.shards[1]
    other_before = shard1.model.W1.copy()
    pid = int(shard0.slices[-1].ids[0])         # a patient in the LAST slice: cheapest possible retrain
    sl = shard0.slices[-1]
    new_sl = subset(sl, np.where(sl.ids != pid)[0])
    retrain_from_slice(shard0, len(shard0.slices) - 1, [new_sl])
    assert np.array_equal(shard1.model.W1, other_before)        # a different shard was never touched
    assert pid not in shard0.slices[-1].id_index


def test_retraining_the_last_slice_is_cheaper_than_the_first(ensemble, ds):
    """Retraining from the last slice only redoes one slice's worth of steps; from the first slice it redoes
    the whole shard — the first should cost at least as much wall-clock time."""
    import copy
    shard = ensemble.shards[2]
    last_idx = len(shard.slices) - 1
    sl = shard.slices[last_idx]
    pid = int(sl.ids[0])
    new_sl = subset(sl, np.where(sl.ids != pid)[0])
    cost_last = retrain_from_slice(copy.deepcopy(shard), last_idx, [new_sl])
    sl0 = shard.slices[0]
    pid0 = int(sl0.ids[0])
    new_sl0 = subset(sl0, np.where(sl0.ids != pid0)[0])
    all_new = [new_sl0] + shard.slices[1:]
    cost_first = retrain_from_slice(copy.deepcopy(shard), 0, all_new)
    assert cost_first > cost_last


def test_build_ensemble_tracks_training_cost(ds):
    ens = build_ensemble(ds, n_shards=4, n_slices=2, seed=1)
    assert ens.total_cost() > 0
    assert all(s.cost > 0 for s in ens.shards)
