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
APIs: PubChem (for drug metadata)
📦 Requirements

Make sure you have the following installed:

🔹 System Requirements
Python 3.8+
Node.js 16+
npm / yarn
Git
🔹 Python Dependencies

Install inside your project:

pip install rdkit-pypi scikit-learn numpy pandas joblib fastapi uvicorn xgboost
🔹 Frontend Dependencies
npm install
▶️ How to Run the Project
⚠️ IMPORTANT

Download/clone the entire project folder before running.

🔹 Step 1 — Train the ML Model
cd "D:\Q-Screen Project\ml"
python train_model.py

This will:

Train models
Generate .pkl files (saved locally, not in repo)
🔹 Step 2 — Start Backend Server
cd "D:\Q-Screen Project\backend"
uvicorn main:app --reload --port 8000

Backend will run at:
👉 http://127.0.0.1:8000

🔹 Step 3 — Start Frontend
cd "D:\Q-Screen Project\frontend"
npm run dev

Frontend will run at:
👉 http://localhost:5173
 (or similar)

🧠 How It Works
Input: User enters molecule (SMILES)
Feature Extraction: RDKit converts structure into numerical form
Prediction: ML model estimates molecular stability
Screening:
Compares with known drugs
Calculates similarity
Output: Displays stability + drug insights
🌍 Real-World Applications
🧬 Drug discovery & screening
🔋 Material and battery research
🌱 Green chemistry
⚛️ Accelerated quantum chemistry workflows
⚠️ Disclaimer

This project is an educational prototype.

Does NOT replace lab experiments
Does NOT perform actual quantum simulations
Does NOT provide medical advice
📁 Project Structure (Simplified)
Q-Screen Project/
│
├── ml/            # Training scripts & models
├── backend/       # FastAPI server
├── frontend/      # React UI
└── README.md
💡 Notes
.pkl model files are not included in the repository
Run training locally to generate them
Ensure correct paths if running on a different system