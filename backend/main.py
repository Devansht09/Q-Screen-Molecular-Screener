"""
Q-Screen FastAPI Backend — Multi-Model
Endpoints: /predict, /predict_all, /models, /similarity, /validate, /visualize, /health
"""

from __future__ import annotations
import os
import sys
import json
import time
import logging
import numpy as np
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(__file__))
from chemistry import (
    parse_smiles, get_descriptors, get_feature_vector,
    energy_to_stability_score, compute_drug_similarity,
    lipinski_check, get_3d_coords, RDKIT_AVAILABLE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger("qscreen")

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Q-Screen API",
    description="Quantum Molecule Stability Screener — Multi-Model ML",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Model Registry ───────────────────────────────────────────────────────────
ML_DIR = os.path.join(os.path.dirname(__file__), "../ml")

MODEL_REGISTRY = {
    "GradientBoosting": {
        "file": "qscreen_model_gradientboosting.pkl",
        "display_name": "Gradient Boosting",
        "description": "Sequential ensemble — each tree corrects the last",
        "color": "#4f9eff",
        "instance": None,
    },
    "XGBoost": {
        "file": "qscreen_model_xgboost.pkl",
        "display_name": "XGBoost",
        "description": "Optimized gradient boosting with regularization",
        "color": "#34d399",
        "instance": None,
    },
    "RandomForest": {
        "file": "qscreen_model_randomforest.pkl",
        "display_name": "Random Forest",
        "description": "Parallel ensemble of independent decision trees",
        "color": "#a78bfa",
        "instance": None,
    },
}

METRICS: dict = {}


def load_models():
    """Load all available trained models and metrics."""
    global METRICS

    # Load metrics JSON if it exists
    metrics_path = os.path.join(ML_DIR, "model_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics_list = json.load(f)
        METRICS = {m["name"]: m for m in metrics_list}
        log.info(f"✓ Loaded metrics for: {list(METRICS.keys())}")
    else:
        log.warning("⚠ model_metrics.json not found — accuracy data unavailable.")

    # Load each model
    try:
        import joblib
    except ImportError:
        log.error("joblib not installed")
        return

    loaded = []
    for key, info in MODEL_REGISTRY.items():
        path = os.path.join(ML_DIR, info["file"])
        if os.path.exists(path):
            try:
                info["instance"] = joblib.load(path)
                loaded.append(key)
                log.info(f"✓ Loaded {key} from {info['file']}")
            except Exception as e:
                log.warning(f"⚠ Could not load {key}: {e}")
        else:
            log.warning(f"⚠ {info['file']} not found — {key} unavailable")

    if not loaded:
        log.warning("⚠ No models loaded — using heuristic fallback for all predictions")
    else:
        log.info(f"✓ Models ready: {loaded}")


load_models()


# ─── Heuristic Fallback ───────────────────────────────────────────────────────
def heuristic_energy(descriptors: dict) -> float:
    n_atoms = descriptors.get("num_atoms", 5)
    hetero = descriptors.get("heteroatoms", 0)
    aromatic = descriptors.get("aromatic_rings", 0)
    logp = descriptors.get("logp", 0)
    rings = descriptors.get("ring_count", 0)
    hbd = descriptors.get("hbd", 0)
    hba = descriptors.get("hba", 0)
    carbon_atoms = max(n_atoms - hetero, 0)
    energy = (
        -38.5 * carbon_atoms + -55.0 * hetero + -25.0 * aromatic +
        -8.0 * rings + -3.0 * abs(logp) + -5.0 * hbd + -3.0 * hba
    )
    return round(energy, 4)


def predict_with_model(model_key: str, features: np.ndarray, descriptors: dict) -> tuple[float, str]:
    """Run prediction with a specific model. Returns (energy_kcal, model_used_str)."""
    info = MODEL_REGISTRY.get(model_key)
    if info and info["instance"] is not None:
        try:
            energy = float(info["instance"].predict(features.reshape(1, -1))[0])
            return energy, model_key
        except Exception as e:
            log.warning(f"{model_key} predict error: {e}")
    return heuristic_energy(descriptors), "heuristic"


# ─── Request / Response Models ────────────────────────────────────────────────
class SMILESInput(BaseModel):
    smiles: str = Field(..., example="CCO")


class SinglePrediction(BaseModel):
    model_name: str
    display_name: str
    description: str
    color: str
    energy_kcal: float
    stability: dict
    mae: Optional[float] = None
    rmse: Optional[float] = None
    r2: Optional[float] = None
    train_time_s: Optional[float] = None
    available: bool


class PredictAllResponse(BaseModel):
    smiles: str
    valid: bool
    descriptors: dict
    lipinski: dict
    predictions: list[SinglePrediction]
    similarity: list[dict]
    processing_ms: float


class VisualizeResponse(BaseModel):
    smiles: str
    sdf: Optional[str]
    available: bool


class ValidateResponse(BaseModel):
    smiles: str
    valid: bool
    canonical: Optional[str]
    formula: Optional[str]
    message: str


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    loaded = [k for k, v in MODEL_REGISTRY.items() if v["instance"] is not None]
    return {
        "service": "Q-Screen API v2",
        "rdkit": RDKIT_AVAILABLE,
        "models_loaded": loaded,
        "endpoints": ["/predict_all", "/models", "/similarity", "/validate", "/visualize", "/health"],
    }


@app.get("/health")
def health():
    loaded = [k for k, v in MODEL_REGISTRY.items() if v["instance"] is not None]
    return {"status": "ok", "rdkit": RDKIT_AVAILABLE, "models_loaded": loaded}


@app.get("/models")
def list_models():
    """Return info about all models including accuracy metrics."""
    result = []
    for key, info in MODEL_REGISTRY.items():
        m = METRICS.get(key, {})
        result.append({
            "key": key,
            "display_name": info["display_name"],
            "description": info["description"],
            "color": info["color"],
            "available": info["instance"] is not None,
            "mae": m.get("mae"),
            "rmse": m.get("rmse"),
            "r2": m.get("r2"),
            "train_time_s": m.get("train_time_s"),
        })
    return {"models": result}


@app.post("/predict_all", response_model=PredictAllResponse)
def predict_all(body: SMILESInput):
    """
    Run all 3 models on the molecule and return results for each.
    This is the main endpoint for the comparison UI.
    """
    t0 = time.perf_counter()
    smiles = body.smiles.strip()

    if not smiles:
        raise HTTPException(400, "SMILES string cannot be empty.")

    mol = parse_smiles(smiles) if RDKIT_AVAILABLE else None
    if RDKIT_AVAILABLE and mol is None:
        raise HTTPException(422, f"Invalid SMILES: '{smiles}'")

    descriptors = get_descriptors(mol) if mol else {
        "molecular_weight": 100.0, "logp": 1.0, "hbd": 0, "hba": 0,
        "tpsa": 20.0, "rotatable_bonds": 0, "aromatic_rings": 0,
        "ring_count": 0, "frac_csp3": 1.0, "heteroatoms": 0,
        "num_atoms": 3, "num_bonds": 2, "heavy_atoms": 3,
        "formula": "C?", "num_stereocenters": 0, "num_amide_bonds": 0,
    }

    lipinski = lipinski_check(descriptors)
    features = get_feature_vector(mol) if mol else None
    similarity = compute_drug_similarity(mol) if mol else []

    predictions = []
    for key, info in MODEL_REGISTRY.items():
        m = METRICS.get(key, {})
        available = info["instance"] is not None

        if features is not None and available:
            energy_kcal, model_used = predict_with_model(key, features, descriptors)
        else:
            energy_kcal = heuristic_energy(descriptors)
            model_used = "heuristic"

        stability = energy_to_stability_score(energy_kcal, mol)

        predictions.append(SinglePrediction(
            model_name=key,
            display_name=info["display_name"],
            description=info["description"],
            color=info["color"],
            energy_kcal=round(energy_kcal, 4),
            stability=stability,
            mae=m.get("mae"),
            rmse=m.get("rmse"),
            r2=m.get("r2"),
            train_time_s=m.get("train_time_s"),
            available=available,
        ))

    elapsed = (time.perf_counter() - t0) * 1000

    return PredictAllResponse(
        smiles=smiles,
        valid=True,
        descriptors=descriptors,
        lipinski=lipinski,
        predictions=predictions,
        similarity=similarity,
        processing_ms=round(elapsed, 2),
    )


# Keep /predict for backward compat — uses GradientBoosting
@app.post("/predict")
def predict(body: SMILESInput):
    result = predict_all(body)
    gb = next((p for p in result.predictions if p.model_name == "GradientBoosting"), result.predictions[0])
    return {
        "smiles": result.smiles,
        "valid": result.valid,
        "descriptors": result.descriptors,
        "energy_kcal": gb.energy_kcal,
        "stability": gb.stability,
        "lipinski": result.lipinski,
        "model_used": gb.display_name,
        "processing_ms": result.processing_ms,
    }


@app.post("/similarity")
def similarity(body: SMILESInput):
    smiles = body.smiles.strip()
    mol = parse_smiles(smiles) if RDKIT_AVAILABLE else None
    if RDKIT_AVAILABLE and mol is None:
        raise HTTPException(422, f"Invalid SMILES: '{smiles}'")
    matches = compute_drug_similarity(mol) or [{"name": "N/A", "smiles": "", "indication": "RDKit required", "similarity": 0.0, "mw": 0}]
    return {"smiles": smiles, "matches": matches}


@app.post("/visualize", response_model=VisualizeResponse)
def visualize(body: SMILESInput):
    smiles = body.smiles.strip()
    sdf = get_3d_coords(smiles)
    return VisualizeResponse(smiles=smiles, sdf=sdf, available=sdf is not None)


@app.post("/validate", response_model=ValidateResponse)
def validate_smiles(body: SMILESInput):
    smiles = body.smiles.strip()
    if not RDKIT_AVAILABLE:
        return ValidateResponse(smiles=smiles, valid=True, canonical=smiles, formula="N/A", message="RDKit unavailable")
    mol = parse_smiles(smiles)
    if mol is None:
        return ValidateResponse(smiles=smiles, valid=False, canonical=None, formula=None, message="Invalid SMILES")
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    canonical = Chem.MolToSmiles(mol)
    formula = rdMolDescriptors.CalcMolFormula(mol)
    return ValidateResponse(smiles=smiles, valid=True, canonical=canonical, formula=formula, message=f"Valid ({formula})")


@app.get("/drugs")
def list_drugs():
    from chemistry import DRUG_DATABASE
    return {"count": len(DRUG_DATABASE), "drugs": DRUG_DATABASE}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
