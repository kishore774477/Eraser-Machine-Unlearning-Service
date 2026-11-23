import pytest
from fastapi.testclient import TestClient

from eraser import server
from eraser.cli import main
from eraser.evaluate import measure_speedup, sweep


# ---------------------------------------------------------------------------- evaluate.py
def test_sweep_shape_and_balance():
    s = sweep(n_seeds=4, n_per_seed=6)
    d = s.to_dict()
    assert d["n_seeds"] == 4
    assert d["loss_ratio_memorized"]["n"] <= 4 * 3 and d["loss_ratio_typical"]["n"] <= 4 * 3
    assert 0.0 <= d["ensemble_acc_before"]["mean"] <= 1.0


def test_measure_speedup_shape():
    r = measure_speedup(n_seeds=3)
    assert r["n_seeds"] == 3 and r["speedup"] > 1.0
    assert r["full_rebuild_seconds"]["mean"] > r["partial_seconds"]["mean"]


def test_the_headline_on_held_out_seeds():
    """The claim this project makes, checked on seeds distinct from the ones used while building it."""
    s = sweep(n_seeds=10, n_per_seed=8, base_seed=9000).to_dict()
    mem, typ = s["loss_ratio_memorized"], s["loss_ratio_typical"]
    assert mem["n"] > 0 and typ["n"] > 0
    assert mem["median"] > typ["median"] * 3
    assert mem["doubled_or_more"] > typ["doubled_or_more"]
    assert typ["median"] == pytest.approx(1.0, abs=0.5)      # barely moves
    assert s["ensemble_acc_after"]["mean"] == pytest.approx(s["ensemble_acc_before"]["mean"], abs=0.03)


# ---------------------------------------------------------------------------- API
@pytest.fixture()
def client():
    app = server.create_app(precompute=False)
    with TestClient(app) as c:
        yield c


def test_meta_and_shards(client):
    m = client.get("/api/meta").json()
    assert m["n_shards"] == 8 and m["n_patients"] == 1600 and m["sweeps_ready"] is False
    sh = client.get("/api/shards").json()
    assert len(sh) == 8 and sum(sum(s["slice_sizes"]) for s in sh) == 1600


def test_patients_pagination_and_lookup(client):
    p = client.get("/api/patients", params={"limit": 20, "offset": 0}).json()
    assert p["total"] == 1600 and len(p["items"]) == 20
    p2 = client.get("/api/patients", params={"limit": 20, "offset": 20}).json()
    assert p["items"][0]["patient_id"] != p2["items"][0]["patient_id"]
    pid = p["items"][0]["patient_id"]
    d = client.get(f"/api/patient/{pid}").json()
    assert d["patient_id"] == pid and "memorized" in d
    assert client.get("/api/patient/999999").status_code == 404


def test_delete_endpoint_and_log(client):
    pid = client.get("/api/patients", params={"limit": 5}).json()["items"][0]["patient_id"]
    r = client.post("/api/delete", json={"patient_id": pid, "full_rebuild": True}).json()
    assert r["patient_id"] == pid and r["speedup"] > 1.0 and "ensemble_accuracy" in r
    log = client.get("/api/log").json()
    assert log[0]["patient_id"] == pid
    assert client.post("/api/delete", json={"patient_id": 999999}).status_code == 404
    assert client.get(f"/api/patient/{pid}").status_code == 404      # really gone


def test_reset_restores_a_fresh_ensemble(client):
    pid = client.get("/api/patients", params={"limit": 1}).json()["items"][0]["patient_id"]
    client.post("/api/delete", json={"patient_id": pid})
    assert client.get(f"/api/patient/{pid}").status_code == 404
    client.post("/api/reset", params={"seed": 1})
    assert client.get(f"/api/patient/{pid}").status_code == 200      # back, because it is a fresh build
    assert client.get("/api/log").json() == []


def test_sweep_endpoint_and_validation(client):
    assert client.get("/api/sweep").json() == {"sweep": None, "speedup": None, "computing": False}
    r = client.post("/api/sweep", params={"n_seeds": 3, "per_seed": 4})
    assert r.status_code == 200 and r.json()["sweep"]["n_seeds"] == 3
    assert client.post("/api/sweep", params={"n_seeds": 1}).status_code == 400
    assert client.post("/api/sweep", params={"per_seed": 1}).status_code == 400


def test_dashboard_is_served(client):
    assert "Eraser" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200 and client.get("/static/style.css").status_code == 200


def test_background_precompute_eventually_fills_the_sweep():
    import time
    app = server.create_app(precompute=True)
    with TestClient(app) as c:
        deadline = time.time() + 60
        ready = False
        while time.time() < deadline:
            if c.get("/api/meta").json()["sweeps_ready"]:
                ready = True
                break
            time.sleep(1)
        assert ready
        assert c.get("/api/sweep").json()["sweep"]["n_seeds"] == 12


# ---------------------------------------------------------------------------- CLI
def test_cli_build(capsys):
    main(["build"])
    out = capsys.readouterr().out
    assert "8 shards x 4 slices" in out and "ensemble accuracy" in out


def test_cli_delete(capsys):
    main(["delete", "1004", "--full-rebuild"])
    out = capsys.readouterr().out
    assert "patient 1004" in out and "x slower" in out


def test_cli_delete_unknown_patient_exits():
    with pytest.raises(SystemExit):
        main(["delete", "999999"])


def test_cli_sweep(capsys):
    main(["sweep", "--seeds", "3", "--per-seed", "4"])
    out = capsys.readouterr().out
    assert "loss_ratio_memorized" in out and "ensemble accuracy" in out


def test_cli_speedup(capsys):
    main(["speedup", "--seeds", "3"])
    out = capsys.readouterr().out
    assert "speedup" in out
