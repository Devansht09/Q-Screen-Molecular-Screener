:::

🧪 Quantum-Inspired Molecular Stability & Drug Screening Platform

An AI-powered platform that predicts molecular stability using machine learning and performs intelligent drug screening by comparing chemical structures with known compounds.

🚀 Overview

This project simulates a real-world drug discovery pipeline by combining:

⚛️ Quantum-inspired molecular stability prediction
🧠 Machine learning-based screening
💊 Drug similarity & therapeutic inference

Instead of running expensive quantum simulations like DFT, this system uses ML models trained on quantum-derived data to deliver fast, scalable predictions, making early-stage research more accessible.

⚙️ Key Features
🔬 Molecular Stability Prediction
Predicts quantum ground-state energy
Filters unstable molecules early
Trained on QM9 dataset
🧠 Drug Screening Module
Identifies known drug matches
Computes Tanimoto similarity
Finds closest structural analogs
💊 Therapeutic Context (Inference-Based)
Suggests potential applications based on similar drugs
Uses database-driven insights (not direct medical prediction)
🌐 Interactive Interface
Input via SMILES strings
Displays:
Stability score
Similarity percentage
Closest known drug
Molecular visualization
🛠️ Tech Stack
Frontend: React + Vite
Backend: FastAPI (Python)
Machine Learning: Scikit-learn (Random Forest, Gradient Boosting, XGBoost)
Cheminformatics: RDKit
Dataset: QM9
APIs: PubChem
📦 Requirements
🔹 System Requirements
Python 3.8+
Node.js 16+
npm
Git
🔹 Python Dependencies
pip install rdkit-pypi scikit-learn numpy pandas joblib fastapi uvicorn xgboost
🔹 Frontend Dependencies
npm install
▶️ How to Run the Project
🔹 Step 1 — Train the ML Model
cd "D:\Q-Screen Project\ml"
python train_model.py
🔹 Step 2 — Start Backend
cd "D:\Q-Screen Project\backend"
uvicorn main:app --reload --port 8000
🔹 Step 3 — Start Frontend
cd "D:\Q-Screen Project\frontend"
npm run dev
🧠 How It Works
Input molecule (SMILES)
Convert to features using RDKit
ML predicts stability
Compare with known drugs
Display results
🌍 Applications
Drug discovery
Material science
Green chemistry
Quantum chemistry acceleration
⚠️ Disclaimer

This is an educational prototype.
Not a substitute for lab testing or medical use.

📁 Structure
ml/
backend/
frontend/