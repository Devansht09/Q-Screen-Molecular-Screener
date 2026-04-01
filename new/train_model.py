"""
Q-Screen ML Training Pipeline — Multi-Model
Trains GradientBoosting, XGBoost, and Random Forest on QM9.
Saves each model as a separate .pkl and writes model_metrics.json.
"""

import os
import json
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import time

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[!] XGBoost not found. Install with: pip install xgboost")

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("[!] RDKit not found. Install with: pip install rdkit")
    exit(1)

HARTREE_TO_KCAL = 627.509474
ML_DIR = os.path.dirname(os.path.abspath(__file__))


# ─── Feature Extraction ───────────────────────────────────────────────────────
def smiles_to_features(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        desc = [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.NumAromaticRings(mol),
            rdMolDescriptors.CalcNumRings(mol),
            Descriptors.FractionCSP3(mol),
            Descriptors.NumHeteroatoms(mol),
            mol.GetNumAtoms(),
            mol.GetNumBonds(),
            Descriptors.MaxPartialCharge(mol),
            Descriptors.MinPartialCharge(mol),
            rdMolDescriptors.CalcNumAmideBonds(mol),
            rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
            rdMolDescriptors.CalcNumSpiroAtoms(mol),
            Descriptors.RingCount(mol),
            Descriptors.NumValenceElectrons(mol),
            Descriptors.HallKierAlpha(mol),
        ]
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        return np.concatenate([desc, np.array(fp)])
    except Exception:
        return None


# ─── XYZ Parser ───────────────────────────────────────────────────────────────
def parse_xyz_file(filepath: str):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        props = lines[1].strip().split('\t')
        u0_kcal = float(props[11]) * HARTREE_TO_KCAL
        smiles = lines[-2].strip().split('\t')[0].strip()
        if not smiles:
            return None
        return smiles, u0_kcal
    except Exception:
        return None


# ─── QM9 Loader ───────────────────────────────────────────────────────────────
def load_qm9_data():
    try:
        import kagglehub
    except ImportError:
        print("[!] kagglehub not found. Install with: pip install kagglehub")
        exit(1)

    print("[+] Downloading QM9 via kagglehub...")
    path = kagglehub.dataset_download("zaharch/quantum-machine-9-aka-qm9")
    print(f"[+] Dataset path: {path}")

    xyz_dir = None
    for root, dirs, files in os.walk(path):
        if any(f.endswith('.xyz') for f in files):
            xyz_dir = root
            break

    if xyz_dir is None:
        print("[!] Could not find XYZ files.")
        exit(1)

    all_files = [f for f in os.listdir(xyz_dir) if f.endswith('.xyz')]
    print(f"[+] Found {len(all_files)} XYZ files. Parsing + featurizing...")

    X, y = [], []
    errors = 0
    for i, fname in enumerate(all_files):
        if i % 10000 == 0 and i > 0:
            print(f"    {i}/{len(all_files)} processed — {len(X)} valid so far...")
        result = parse_xyz_file(os.path.join(xyz_dir, fname))
        if result is None:
            errors += 1
            continue
        smiles, u0_kcal = result
        features = smiles_to_features(smiles)
        if features is not None:
            X.append(features)
            y.append(u0_kcal)
        else:
            errors += 1

    print(f"[+] Parsed {len(X)} molecules ({errors} skipped).")
    return np.array(X), np.array(y)


# ─── Train & Evaluate One Model ───────────────────────────────────────────────
def train_and_evaluate(name, pipeline, X_train, X_test, y_train, y_test):
    print(f"\n[+] Training {name}...")
    t0 = time.time()
    pipeline.fit(X_train, y_train)
    train_time = round(time.time() - t0, 1)

    y_pred = pipeline.predict(X_test)
    mae  = round(float(mean_absolute_error(y_test, y_pred)), 4)
    rmse = round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4)
    r2   = round(float(r2_score(y_test, y_pred)), 4)

    print(f"    MAE  : {mae} kcal/mol")
    print(f"    RMSE : {rmse} kcal/mol")
    print(f"    R²   : {r2}")
    print(f"    Time : {train_time}s")

    return pipeline, {"name": name, "mae": mae, "rmse": rmse, "r2": r2, "train_time_s": train_time}


# ─── Main ─────────────────────────────────────────────────────────────────────
def train():
    print("=" * 60)
    print("  Q-Screen Multi-Model Training Pipeline")
    print("=" * 60)

    if not XGBOOST_AVAILABLE:
        print("[!] XGBoost not available. Install with: pip install xgboost")
        print("    Continuing with GradientBoosting and RandomForest only.")

    X, y = load_qm9_data()
    print(f"[+] Feature matrix : {X.shape}")
    print(f"[+] Energy range   : {y.min():.1f} to {y.max():.1f} kcal/mol")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    print(f"[+] Train: {len(X_train)} | Test: {len(X_test)}")

    # ── Model definitions ──
    models = [
        (
            "GradientBoosting",
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", GradientBoostingRegressor(
                    n_estimators=200, max_depth=5, learning_rate=0.08,
                    subsample=0.85, min_samples_leaf=10, random_state=42, verbose=0,
                )),
            ]),
        ),
        (
            "RandomForest",
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", RandomForestRegressor(
                    n_estimators=300, max_depth=14, min_samples_leaf=5,
                    n_jobs=-1, random_state=42, verbose=0,
                )),
            ]),
        ),
    ]

    if XGBOOST_AVAILABLE:
        models.append((
            "XGBoost",
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", XGBRegressor(
                    n_estimators=300, max_depth=6, learning_rate=0.08,
                    subsample=0.85, colsample_bytree=0.8, min_child_weight=5,
                    random_state=42, verbosity=0, n_jobs=-1,
                )),
            ]),
        ))

    # ── Train all models ──
    all_metrics = []
    for name, pipeline in models:
        trained_pipeline, metrics = train_and_evaluate(
            name, pipeline, X_train, X_test, y_train, y_test
        )
        # Save model
        model_filename = f"qscreen_model_{name.lower()}.pkl"
        model_path = os.path.join(ML_DIR, model_filename)
        joblib.dump(trained_pipeline, model_path)
        metrics["file"] = model_filename
        all_metrics.append(metrics)
        print(f"[+] Saved → {model_path}")

    # Also save GradientBoosting as the default qscreen_model.pkl for backward compat
    gb_pipeline = next(p for (n, p) in models if n == "GradientBoosting")
    joblib.dump(gb_pipeline, os.path.join(ML_DIR, "qscreen_model.pkl"))
    print(f"[+] Default model saved → qscreen_model.pkl (GradientBoosting)")

    # Save metrics JSON
    metrics_path = os.path.join(ML_DIR, "model_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"[+] Metrics saved → {metrics_path}")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  TRAINING SUMMARY")
    print("=" * 60)
    print(f"{'Model':<22} {'MAE':>10} {'RMSE':>10} {'R²':>8} {'Time':>8}")
    print("-" * 60)
    for m in sorted(all_metrics, key=lambda x: x["mae"]):
        print(f"{m['name']:<22} {m['mae']:>10} {m['rmse']:>10} {m['r2']:>8} {m['train_time_s']:>7}s")
    best = min(all_metrics, key=lambda x: x["mae"])
    print(f"\n  Best model by MAE: {best['name']} ({best['mae']} kcal/mol)")
    print("=" * 60)
    print("  Next: cd backend && uvicorn main:app --reload --port 8000")
    print("=" * 60)


if __name__ == "__main__":
    train()
