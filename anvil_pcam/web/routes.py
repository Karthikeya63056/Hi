"""Anvil PCAM API routes and HTML endpoint."""

from __future__ import annotations

import time

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse

from anvil_pcam.core import NOISE_PROFILES, benchmark_bank, evaluate_trial, predict_precision, precision_summary

router = APIRouter()


def _engine(request: Request):
    return request.app.state.pcam


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return request.app.state.templates.TemplateResponse(request, "index.html")


@router.get("/api/pcam/state")
async def pcam_state(request: Request):
    engine = _engine(request)
    return {
        "dimension": 64,
        "patterns": [pattern.as_dict() for pattern in engine.store.patterns],
        "graph": engine.store.graph(),
        "config": {
            "beta": engine.config.beta,
            "iterations": engine.config.iterations,
            "stepSize": engine.config.step_size,
        },
        "noiseProfiles": NOISE_PROFILES,
        "terminology": [
            "Associative Memory",
            "Memory Attractor",
            "Precision Operator Π",
            "Energy Dynamics",
            "Anisotropic Precision",
            "Inference-Time Steering",
        ],
    }


@router.post("/api/pcam/trial")
async def pcam_trial(request: Request):
    engine = _engine(request)
    body = await request.json()
    pattern_id = body.get("pattern_id") or engine.store.patterns[0].id
    gaussian_sigma = float(body.get("gaussian_sigma", 0.58))
    mask_fraction = float(body.get("mask_fraction", 0.28))
    profile = body.get("profile", "demo")
    seed = body.get("seed")
    if seed is None:
        seed = int(time.time() * 1000) % 1_000_000

    try:
        return evaluate_trial(
            engine,
            pattern_id=pattern_id,
            gaussian_sigma=gaussian_sigma,
            mask_fraction=mask_fraction,
            seed=int(seed),
            profile=profile,
        )
    except KeyError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


@router.post("/api/pcam/precision")
async def pcam_precision(request: Request):
    body = await request.json()
    precision = predict_precision(body["corrupted_query"])
    return {
        "precision": precision.round(5).tolist(),
        "summary": precision_summary(precision),
    }


@router.get("/api/pcam/benchmark")
async def pcam_benchmark(request: Request, seed: int = 7, profile: str = "stress"):
    return benchmark_bank(_engine(request), seed=seed, profile=profile)
