"""
Q-Screen Chemistry Engine
Handles: SMILES parsing, descriptor computation, fingerprints, similarity
"""

from __future__ import annotations
import numpy as np
from typing import Optional

try:
    from rdkit import Chem
    from rdkit.Chem import (
        Descriptors, AllChem, rdMolDescriptors,
        Draw, rdDepictor, rdmolops
    )
    from rdkit.DataStructs import TanimotoSimilarity
    from rdkit import DataStructs
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

# ─── Known Drug Database ──────────────────────────────────────────────────────
# Curated set of FDA-approved drugs with SMILES, name, and indication.
DRUG_DATABASE = [
    {
        "name": "Aspirin",
        "smiles": "CC(=O)Oc1ccccc1C(=O)O",
        "indication": "Anti-inflammatory / Analgesic",
        "mw": 180.16,
    },
    {
        "name": "Ibuprofen",
        "smiles": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "indication": "NSAID / Pain relief",
        "mw": 206.28,
    },
    {
        "name": "Paracetamol (Acetaminophen)",
        "smiles": "CC(=O)Nc1ccc(O)cc1",
        "indication": "Analgesic / Antipyretic",
        "mw": 151.16,
    },
    {
        "name": "Caffeine",
        "smiles": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
        "indication": "CNS Stimulant",
        "mw": 194.19,
    },
    {
        "name": "Metformin",
        "smiles": "CN(C)C(=N)NC(=N)N",
        "indication": "Type 2 Diabetes",
        "mw": 129.16,
    },
    {
        "name": "Sildenafil (Viagra)",
        "smiles": "CCCC1=NN(C2=CC(=CC=C12)S(=O)(=O)N3CCN(CC3)C)C4=NC(=O)C5=C(N4)C=CC(=C5)OCC",
        "indication": "Erectile Dysfunction / Pulmonary HTN",
        "mw": 474.58,
    },
    {
        "name": "Atorvastatin (Lipitor)",
        "smiles": "CC(C)c1n(CC(O)CC(O)CC(=O)O)c2cc(ccc2c1C(=O)Nc1ccccc1F)c1ccccc1",
        "indication": "Hypercholesterolemia",
        "mw": 558.64,
    },
    {
        "name": "Omeprazole",
        "smiles": "COc1ccc2nc([nH]c2c1)S(=O)Cc1ncc(C)c(OC)c1C",
        "indication": "GERD / Peptic Ulcer",
        "mw": 345.42,
    },
    {
        "name": "Amoxicillin",
        "smiles": "CC1(C)SC2C(NC(=O)C(N)c3ccc(O)cc3)C(=O)N2C1C(=O)O",
        "indication": "Antibiotic (β-lactam)",
        "mw": 365.40,
    },
    {
        "name": "Ciprofloxacin",
        "smiles": "O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O",
        "indication": "Antibiotic (Fluoroquinolone)",
        "mw": 331.34,
    },
    {
        "name": "Lisinopril",
        "smiles": "NCCCC(NC(CCc1ccccc1)C(=O)O)C(=O)N1CCCC1C(=O)O",
        "indication": "ACE Inhibitor / Hypertension",
        "mw": 405.49,
    },
    {
        "name": "Fluoxetine (Prozac)",
        "smiles": "CNCCC(Oc1ccc(cc1)C(F)(F)F)c1ccccc1",
        "indication": "SSRI / Depression",
        "mw": 309.33,
    },
    {
        "name": "Diazepam (Valium)",
        "smiles": "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21",
        "indication": "Anxiolytic / Benzodiazepine",
        "mw": 284.74,
    },
    {
        "name": "Warfarin",
        "smiles": "OC(=O)Cc1ccccc1OC(=O)c1ccc(Cl)cc1",
        "indication": "Anticoagulant",
        "mw": 308.33,
    },
    {
        "name": "Losartan",
        "smiles": "CCCCc1nc(Cl)c(CO)n1Cc1ccc(-c2ccccc2-c2nnn[nH]2)cc1",
        "indication": "Angiotensin II Blocker / HTN",
        "mw": 422.91,
    },
    {
        "name": "Tamoxifen",
        "smiles": "CCC(=C(c1ccccc1)c1ccc(OCCN(C)C)cc1)c1ccccc1",
        "indication": "Breast Cancer / ER antagonist",
        "mw": 371.51,
    },
    {
        "name": "Doxorubicin",
        "smiles": "COc1cccc2C(=O)c3c(O)c4CC(O)(C(=O)CO)Cc4c(O)c3C(=O)c12",
        "indication": "Chemotherapy / Anthracycline",
        "mw": 543.52,
    },
    {
        "name": "Oseltamivir (Tamiflu)",
        "smiles": "CCOC(=O)C1=CC(CC(CC1)OC(=O)C(C)C)N",
        "indication": "Antiviral / Influenza",
        "mw": 312.40,
    },
    {
        "name": "Chloroquine",
        "smiles": "CCN(CC)CCCC(C)Nc1ccnc2cc(Cl)ccc12",
        "indication": "Antimalarial / Anti-inflammatory",
        "mw": 319.87,
    },
]


def parse_smiles(smiles: str) -> Optional[object]:
    """Parse SMILES → RDKit mol or None if invalid."""
    if not RDKIT_AVAILABLE:
        return None
    mol = Chem.MolFromSmiles(smiles.strip())
    return mol


def get_descriptors(mol) -> dict:
    """Compute molecular descriptors from RDKit mol object."""
    if not RDKIT_AVAILABLE or mol is None:
        return {}

    return {
        "molecular_weight": round(Descriptors.MolWt(mol), 3),
        "logp": round(Descriptors.MolLogP(mol), 3),
        "hbd": int(Descriptors.NumHDonors(mol)),
        "hba": int(Descriptors.NumHAcceptors(mol)),
        "tpsa": round(Descriptors.TPSA(mol), 3),
        "rotatable_bonds": int(Descriptors.NumRotatableBonds(mol)),
        "aromatic_rings": int(Descriptors.NumAromaticRings(mol)),
        "ring_count": int(rdMolDescriptors.CalcNumRings(mol)),
        "frac_csp3": round(Descriptors.FractionCSP3(mol), 4),
        "heteroatoms": int(Descriptors.NumHeteroatoms(mol)),
        "num_atoms": mol.GetNumAtoms(),
        "num_bonds": mol.GetNumBonds(),
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "num_stereocenters": len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
        "num_amide_bonds": int(rdMolDescriptors.CalcNumAmideBonds(mol)),
    }


def get_feature_vector(mol) -> Optional[np.ndarray]:
    """Generate feature vector (descriptors + Morgan FP) for ML model."""
    if not RDKIT_AVAILABLE or mol is None:
        return None

    # Must match the descriptor list used in `ml/train_model.py`.
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
        mol.GetNumHeavyAtoms(),
    ]

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    fp_arr = np.zeros((2048,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, fp_arr)

    return np.concatenate([np.asarray(desc, dtype=np.float32), fp_arr.astype(np.float32)])


def energy_to_stability_score(energy_kcal: float, mol) -> dict:
    n_atoms = mol.GetNumHeavyAtoms() if mol else 1
    energy_per_atom = energy_kcal / max(n_atoms, 1)

    # Wider range to handle heuristic values (-15 to -60 kcal/mol per atom)
    raw_score = (energy_per_atom - (-15)) / (-60 - (-15)) * 100
    score = float(np.clip(raw_score, 0, 100))

    if score >= 75:
        label, color = "Highly Stable", "#00ff9d"
    elif score >= 50:
        label, color = "Moderately Stable", "#a3ff00"
    elif score >= 25:
        label, color = "Marginally Stable", "#ffd700"
    else:
        label, color = "Unstable", "#ff4757"

    return {
        "score": round(score, 2),
        "label": label,
        "color": color,
        "energy_per_atom": round(energy_per_atom, 4),
    }  
    """
    Convert predicted ground-state energy to a 0-100 stability score.
    Lower (more negative) energy = more stable.
    Rough empirical normalization based on QM9 distribution.
    """
    # QM9 targets are typically in Hartree (Eh). We normalize per heavy atom.
    n_atoms = mol.GetNumHeavyAtoms() if mol else 1
    energy_per_atom = energy_kcal / max(n_atoms, 1)

    # QM9 U0 (Hartree) per heavy atom is roughly in a comparable numeric band
    # (e.g., methane ~ -40 Eh for 1 heavy atom). We map:
    # -50 → 100 (very stable), -20 → 0 (unstable), linear
    raw_score = (energy_per_atom - (-20)) / (-50 - (-20)) * 100
    score = float(np.clip(raw_score, 0, 100))

    if score >= 75:
        label, color = "Highly Stable", "#00ff9d"
    elif score >= 50:
        label, color = "Moderately Stable", "#a3ff00"
    elif score >= 25:
        label, color = "Marginally Stable", "#ffd700"
    else:
        label, color = "Unstable", "#ff4757"

    return {
        "score": round(score, 2),
        "label": label,
        "color": color,
        "energy_per_atom": round(energy_per_atom, 4),
    }


def get_morgan_fp(mol):
    """Return Morgan fingerprint object for Tanimoto comparison."""
    if not RDKIT_AVAILABLE or mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)


def compute_drug_similarity(query_mol) -> list[dict]:
    """
    Compare query molecule to drug database via Tanimoto similarity.
    Returns top 5 matches sorted by similarity.
    """
    if not RDKIT_AVAILABLE or query_mol is None:
        return []

    query_fp = get_morgan_fp(query_mol)
    if query_fp is None:
        return []

    results = []
    for drug in DRUG_DATABASE:
        ref_mol = Chem.MolFromSmiles(drug["smiles"])
        if ref_mol is None:
            continue
        ref_fp = get_morgan_fp(ref_mol)
        sim = TanimotoSimilarity(query_fp, ref_fp)
        results.append({
            "name": drug["name"],
            "smiles": drug["smiles"],
            "indication": drug["indication"],
            "similarity": round(sim * 100, 2),
            "mw": drug["mw"],
        })

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:5]


def lipinski_check(descriptors: dict) -> dict:
    """Evaluate Lipinski Rule of Five for drug-likeness."""
    checks = {
        "MW ≤ 500 Da": descriptors.get("molecular_weight", 999) <= 500,
        "LogP ≤ 5": descriptors.get("logp", 99) <= 5,
        "HBD ≤ 5": descriptors.get("hbd", 99) <= 5,
        "HBA ≤ 10": descriptors.get("hba", 99) <= 10,
    }
    violations = sum(1 for v in checks.values() if not v)
    return {
        "rules": checks,
        "violations": violations,
        "drug_like": violations <= 1,
        "verdict": "Drug-like ✓" if violations <= 1 else f"Fails Ro5 ({violations} violation{'s' if violations > 1 else ''})",
    }


def get_3d_coords(smiles: str) -> Optional[str]:
    """Generate 3D SDF for 3Dmol.js visualization."""
    if not RDKIT_AVAILABLE:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        AllChem.MMFFOptimizeMolecule(mol)
        mol = Chem.RemoveHs(mol)
        return Chem.MolToMolBlock(mol)
    except Exception:
        return None
