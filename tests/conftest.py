import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["ERASER_SEED"] = "1"

from eraser.data import held_out, make_dataset, make_task   # noqa: E402
from eraser.sisa import build_ensemble                        # noqa: E402


@pytest.fixture(scope="session")
def task():
    return make_task(1)


@pytest.fixture(scope="session")
def ds(task):
    return make_dataset(task, 1600, seed=1)


@pytest.fixture(scope="session")
def test_ds(task):
    return held_out(task, seed=1, n=400)


@pytest.fixture()
def ensemble(ds):
    """A fresh ensemble per test (deletion mutates it), built from the shared dataset/task."""
    return build_ensemble(ds, n_shards=8, n_slices=4, seed=1)
