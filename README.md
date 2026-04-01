# 🧪 Quantum-Inspired Molecular Stability & Drug Screening Platform

An AI-powered platform that predicts molecular stability and performs drug screening.

---

## 🚀 Overview

This project simulates a simplified **drug discovery pipeline** by combining:

- Molecular stability prediction using ML  
- Drug similarity screening  
- Basic therapeutic inference  

Instead of expensive quantum simulations, it uses trained ML models to give fast predictions.

---

## ⚙️ Key Features

### 🔬 Stability Prediction
- Predicts molecular energy (stability)
- Filters unstable molecules

### 🧠 Drug Screening
- Finds similar known drugs
- Uses Tanimoto similarity

### 💊 Inference
- Suggests possible usage based on similar drugs

---

## 📦 Requirements

Make sure you have:

- Python 3.8+  
- Node.js 16+  
- npm  
- Git  

---

## 🛠️ Installation

### Install Python dependencies

```bash
pip install rdkit-pypi scikit-learn numpy pandas joblib fastapi uvicorn xgboost
```

---

### Install frontend dependencies

```bash
npm install
```

---

## 🚀 How to Run

### 1. Train Model
```bash
cd "D:\Q-Screen Project\ml"
python train_model.py
```

### 2. Run Backend
```bash
cd "D:\Q-Screen Project\backend"
uvicorn main:app --reload --port 8000
```

### 3. Run Frontend
```bash
cd "D:\Q-Screen Project\frontend"
npm run dev
```

---

## ⚙️ Tech Stack
- Python (FastAPI)
- React + Vite
- Scikit-learn
- RDKit