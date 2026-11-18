"""FastAPI backend — runs on Laptop 1.  Owner: P4

Run:  uvicorn api.main:app --host 0.0.0.0 --port 8000
Demo: PATHSNAP_EVAL_DIR=/path/to/evaluator_trips uvicorn api.main:app --host 0.0.0.0 --port 8000

Every request is logged to the terminal (evaluators must see these logs on Laptop 1).
"""
from __future__ import annotations

import logging
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from pathsnap.config import CFG
from pathsnap.io import list_trips, load_trip

from .schemas import Health, PredictRequest, PredictResponse, TripInfo, TripList

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("pathsnap.api")

EVAL_DIR = CFG["paths"]["eval_dir"]
app = FastAPI(title="PathSnap API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

predictor = None  # PathSnapPredictor, loaded at startup


@app.on_event("startup")
def _load():
    global predictor
    try:
        from pathsnap.pipeline import PathSnapPredictor

        predictor = PathSnapPredictor(CFG["paths"]["artifacts_dir"])
        log.info("Model loaded from %s", CFG["paths"]["artifacts_dir"])
    except NotImplementedError:
        log.warning("Pipeline not implemented yet — /predict will return 503")
    log.info("Serving trips from %s", EVAL_DIR)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    log.info("%s %s from %s -> %d (%.0f ms)", request.method, request.url.path,
             request.client.host if request.client else "?", response.status_code,
             (time.perf_counter() - t0) * 1000)
    return response


@app.get("/health", response_model=Health)
def health():
    return Health(status="ok", model_name=getattr(predictor, "model_name", None),
                  eval_dir=EVAL_DIR, n_trips=len(list_trips(EVAL_DIR)))


@app.get("/trips", response_model=TripList)
def trips():
    out = []
    for tid in list_trips(EVAL_DIR):
        trip = load_trip(EVAL_DIR, tid, with_route=False)
        out.append(TripInfo(trip_id=tid, n_points=len(trip.track)))
    return TripList(partition=EVAL_DIR, trips=out)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if req.trip_id not in list_trips(EVAL_DIR):
        raise HTTPException(404, f"unknown trip {req.trip_id}")
    if predictor is None:
        raise HTTPException(503, "model not loaded")
    log.info("Predicting trip %s", req.trip_id)
    # TODO(P4): result, net = predictor.predict(EVAL_DIR, req.trip_id)
    #           metrics = evaluate.evaluate_prediction(result, EVAL_DIR, net)   # server-side only
    #           build PredictResponse (lat/lon polylines via net.arc_polyline_lonlat), log metrics
    raise HTTPException(501, "not implemented")
