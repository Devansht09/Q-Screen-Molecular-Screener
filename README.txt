# Q-Screen — Quantum Molecule Stability Screener

> ML-powered molecular stability prediction and drug similarity screening in your browser.

![Q-Screen Demo](https://img.shields.io/badge/status-ready-00ff9d?style=for-the-badge)

---

## What It Does

Q-Screen lets you enter any molecule as a **SMILES string** and instantly get:

| Feature | Description |
|---|---|
| 🧪 **Stability Score** | ML-predicted ground-state energy → 0–100 stability index |
| 💊 **Drug Match** | Tanimoto similarity vs. 20 FDA-approved drugs |
| 🌐 **3D Viewer** | Real-time 3D structure via 3Dmol.js |
| 📊 **Descriptors** | MolWt, LogP, TPSA, HBD/A, Lipinski Ro5, and more |

---

## Quick Start

### 1. Clone / Enter Project

```bash
cd qscreen
```

### 2. Python Environment

```bash
# Option A: Conda (recommended for RDKit)
conda create -n qscreen python=3.10
conda activate qscreen
conda install -c conda-forge rdkit
pip install fastapi uvicorn scikit-learn pandas numpy joblib

# Option B: pip only
pip install rdkit fastapi uvicorn scikit-learn pandas numpy joblib
```

### 3. Train the ML Model

```bash
cd ml
python train_model.py
# → Saves: ml/qscreen_model.pkl
```

If you have a QM9 subset CSV, place it at `data/qm9_subset.csv` with columns:
- `smiles` — SMILES notation
- `u0` — internal energy at 0K (kcal/mol)

Otherwise the trainer uses the built-in curated SMILES library.

### 4. Start the Backend

```bash
cd backend
uvicorn main:app --reload --port 8000
# API docs at: http://localhost:8000/docs
```

### 5. Start the Frontend

```bash
cd frontend
npm install
npm run dev
# Opens at: http://localhost:3000
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/validate` | Validate & canonicalize SMILES |
| `POST` | `/predict` | Full stability analysis |
| `POST` | `/similarity` | Drug similarity matching |
| `POST` | `/visualize` | 3D SDF coordinates |
| `GET` | `/drugs` | List drug reference database |
| `GET` | `/health` | Health check |

### Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"smiles": "CCO"}'
```

```json
{
  "smiles": "CCO",
  "valid": true,
  "descriptors": {
    "molecular_weight": 46.069,
    "logp": -0.001,
    "hbd": 1,
    "hba": 1,
    "tpsa": 20.23,
    "formula": "C2H6O"
  },
  "energy_kcal": -154.05,
  "stability": {
    "score": 72.4,
    "label": "Moderately Stable",
    "color": "#a3ff00",
    "energy_per_atom": -22.01
  },
  "lipinski": {
    "drug_like": true,
    "verdict": "Drug-like ✓"
  },
  "model_used": "GradientBoostingRegressor",
  "processing_ms": 3.2
}
```

---

## Architecture

```
qscreen/
├── backend/
│   ├── main.py          ← FastAPI app + all endpoints
│   ├── chemistry.py     ← RDKit engine: descriptors, fingerprints, similarity
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx      ← Main React application
│   │   ├── index.css    ← Dark lab aesthetic styles
│   │   └── main.jsx     ← Entry point
│   ├── index.html       ← 3Dmol.js CDN loaded here
│   ├── vite.config.js
│   └── package.json
├── ml/
│   ├── train_model.py   ← Training pipeline (GradientBoostingRegressor)
│   └── qscreen_model.pkl  ← Saved after training
└── data/
    └── qm9_subset.csv   ← Optional: place QM9 data here
```

---

## ML Model Details

**Features (2068 total):**
- 20 RDKit molecular descriptors (MolWt, LogP, TPSA, HBD, HBA, rings, etc.)
- 2048-bit Morgan fingerprint (radius=2)

**Model:** `GradientBoostingRegressor`
- 200 estimators, max_depth=5, learning_rate=0.08
- Scikit-learn Pipeline with StandardScaler

**Target:** Ground-state internal energy U₀ (kcal/mol)

**Stability Score:** Per-atom energy normalized to [0, 100] based on QM9 distribution.

---

## Drug Database

20 curated FDA-approved drugs included:
Aspirin, Ibuprofen, Paracetamol, Caffeine, Metformin, Sildenafil, Atorvastatin, Omeprazole, Amoxicillin, Ciprofloxacin, Lisinopril, Fluoxetine, Diazepam, Warfarin, Morphine, Losartan, Tamoxifen, Doxorubicin, Oseltamivir, Chloroquine

---

## Disclaimers

- **Not clinically validated** — do not use for medical decisions
- **Not real DFT** — ML approximation of quantum mechanics
- **Not replacing pharmaceutical pipelines** — pre-screening research tool

---

## License

MIT — Build freely, cite responsibly.
