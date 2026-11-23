"""FastAPI app: the SISA ensemble, the deletion/audit API, and the dashboard."""
from __future__ import annotations

import os
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import __version__
from .audit import audit_and_delete, memorization_risk
from .data import held_out, make_dataset, make_task, row_of
from .evaluate import measure_speedup, sweep
from .model import accuracy
from .sisa import build_ensemble

ROOT = Path(__file__).resolve().parent.parent
N_SHARDS, N_SLICES, N_PATIENTS = 8, 4, 1600


class DeleteReq(BaseModel):
    patient_id: int
    full_rebuild: bool = False


def create_app(precompute: bool = True) -> FastAPI:
    state: dict = {"sweep": None, "speedup": None, "computing": False, "log": []}

    def build(seed: int) -> None:
        task = make_task(seed)
        ds = make_dataset(task, N_PATIENTS, seed=seed)
        test = held_out(task, seed=seed, n=400)
        ens = build_ensemble(ds, N_SHARDS, N_SLICES, seed=seed)
        state.update(seed=seed, task=task, ds=ds, test=test, ens=ens, log=[])

    build(int(os.getenv("ERASER_SEED", "1")))

    if precompute:
        def work():
            state["computing"] = True
            try:
                state["sweep"] = sweep(12, 8).to_dict()
                state["speedup"] = measure_speedup(8)
            finally:
                state["computing"] = False
        threading.Thread(target=work, daemon=True).start()

    app = FastAPI(title="Eraser", version=__version__, description="Machine unlearning: SISA sharded training, a deletion API, and a membership-inference audit.")
    app.state.get = lambda: state

    @app.get("/api/meta")
    def meta():
        ens = state["ens"]
        return {"version": __version__, "seed": state["seed"], "n_shards": len(ens.shards), "n_slices": N_SLICES, "n_patients": N_PATIENTS,
                "ensemble_accuracy": round(accuracy(ens.predict_proba(state["test"].X), state["test"].y), 4), "sweeps_ready": state["sweep"] is not None}

    @app.get("/api/shards")
    def shards():
        ens = state["ens"]
        return [{"index": s.index, "n_slices": len(s.slices), "slice_sizes": [len(sl.y) for sl in s.slices], "cost": round(s.cost, 4)} for s in ens.shards]

    @app.get("/api/patients")
    def patients(limit: int = 60, offset: int = 0):
        ds = state["ds"]
        ens = state["ens"]
        ids = ds.ids[offset:offset + min(limit, 200)]
        out = []
        for pid in ids:
            pid = int(pid)
            shard_idx, slice_idx = ens.locate(pid)
            out.append({"patient_id": pid, "shard": shard_idx, "slice": slice_idx, "memorized": memorization_risk(ens, ds, pid)})
        return {"total": len(ds.ids), "items": out}

    @app.get("/api/patient/{patient_id}")
    def patient(patient_id: int):
        ds, ens = state["ds"], state["ens"]
        try:
            row = row_of(ds, patient_id)
            shard_idx, slice_idx = ens.locate(patient_id)
        except KeyError as e:
            raise HTTPException(404, str(e))
        from .model import per_example_loss, predict_proba
        loss = float(per_example_loss(predict_proba(ens.shards[shard_idx].model, ds.X[row:row + 1]), ds.y[row:row + 1])[0])
        return {"patient_id": patient_id, "shard": shard_idx, "slice": slice_idx, "memorized": memorization_risk(ens, ds, patient_id),
                "current_loss": round(loss, 5), "label": int(ds.y[row])}

    @app.post("/api/delete")
    def delete(req: DeleteReq):
        ds, ens, test = state["ds"], state["ens"], state["test"]
        try:
            r = audit_and_delete(ens, ds, test, req.patient_id, measure_full_rebuild=req.full_rebuild)
        except KeyError as e:
            raise HTTPException(404, str(e))
        d = r.to_dict()
        state["log"].insert(0, d)
        state["log"] = state["log"][:30]
        d["ensemble_accuracy"] = round(accuracy(ens.predict_proba(test.X), test.y), 4)
        return d

    @app.get("/api/log")
    def log():
        return state["log"]

    @app.post("/api/reset")
    def reset(seed: int = 1):
        build(seed)
        return {"seed": seed}

    @app.get("/api/sweep")
    def sweep_():
        return {"sweep": state["sweep"], "speedup": state["speedup"], "computing": state["computing"]}

    @app.post("/api/sweep")
    def sweep_run(n_seeds: int = 12, per_seed: int = 8):
        if not 2 <= n_seeds <= 60:
            raise HTTPException(400, "n_seeds must be between 2 and 60")
        if not 2 <= per_seed <= 40:
            raise HTTPException(400, "per_seed must be between 2 and 40")
        state["sweep"] = sweep(n_seeds, per_seed).to_dict()
        state["speedup"] = measure_speedup(min(n_seeds, 10))
        return {"sweep": state["sweep"], "speedup": state["speedup"]}

    @app.get("/")
    def index():
        return FileResponse(ROOT / "web" / "index.html")

    app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")
    return app
