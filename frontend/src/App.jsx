import { useState, useEffect, useRef, useCallback } from "react";

const API = "http://localhost:8000";

const api = async (endpoint, body) => {
  const r = await fetch(`${API}${endpoint}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(err.detail || "API Error");
  }
  return r.json();
};

const apiGet = async (endpoint) => {
  const r = await fetch(`${API}${endpoint}`);
  if (!r.ok) throw new Error(r.statusText);
  return r.json();
};

// ─── Atom Loader ──────────────────────────────────────────────────────────────
function AtomLoader({ message }) {
  return (
    <div className="atom-overlay">
      <div className="atom-scene">
        <div className="atom-core" />
        <div className="orbit-ring r1"><div className="orbit-dot" /></div>
        <div className="orbit-ring r2"><div className="orbit-dot" /></div>
        <div className="orbit-ring r3"><div className="orbit-dot" /></div>
      </div>
      <div className="loader-label">Analysing Structure</div>
      <div className="loader-msg">{message}</div>
      <div className="loader-dots">
        <div className="ld" /><div className="ld" /><div className="ld" />
      </div>
    </div>
  );
}

// ─── Score Orb ────────────────────────────────────────────────────────────────
function ScoreOrb({ score, label, color, size = 220 }) {
  const [animated, setAnimated] = useState(0);
  useEffect(() => {
    const t = setTimeout(() => setAnimated(score), 150);
    return () => clearTimeout(t);
  }, [score]);

  const r = (size / 2) - 14;
  const circ = 2 * Math.PI * r;
  const dash = (animated / 100) * circ;
  const cx = size / 2, cy = size / 2;

  return (
    <div className="score-orb-wrap">
      <div className="score-orb" style={{ width: size, height: size }}>
        <svg className="score-ring-svg" viewBox={`0 0 ${size+28} ${size+28}`}>
          <circle cx={cx+14} cy={cy+14} r={r} fill="none" stroke="rgba(255,255,255,0.04)" strokeWidth="8"/>
          <circle cx={cx+14} cy={cy+14} r={r} fill="none" stroke={color}
            strokeWidth="8" strokeLinecap="round"
            transform={`rotate(-90 ${cx+14} ${cy+14})`}
            strokeDasharray={`${dash} ${circ}`}
            style={{ transition: "stroke-dasharray 1.5s cubic-bezier(.22,1,.36,1)", filter: `drop-shadow(0 0 12px ${color})` }}
          />
        </svg>
        <div className="score-num" style={{ color, fontSize: size > 160 ? "3.6rem" : "2.2rem" }}>
          {Math.round(animated)}
        </div>
        <div className="score-100">/ 100</div>
        <div className="score-badge" style={{ color, borderColor: color + "44", background: color + "18" }}>
          {label}
        </div>
      </div>
    </div>
  );
}

// ─── Sat Card ─────────────────────────────────────────────────────────────────
function SatCard({ label, value, unit, sub }) {
  return (
    <div className="sat-card">
      <div className="sc-lbl">{label}</div>
      <div className="sc-val">{value}</div>
      {unit && <div className="sc-unit">{unit}</div>}
      {sub && <div className="sc-sub">{sub}</div>}
    </div>
  );
}

// ─── Model Card ───────────────────────────────────────────────────────────────
function ModelCard({ prediction, isActive, onClick, rank }) {
  const [animated, setAnimated] = useState(0);
  useEffect(() => {
    const t = setTimeout(() => setAnimated(prediction.stability.score), 300);
    return () => clearTimeout(t);
  }, [prediction.stability.score]);

  const rankLabels = ["#1 Best", "#2", "#3"];
  const rankColors = ["#fbbf24", "#94a3b8", "#cd7f32"];

  return (
    <div
      className={`model-card ${isActive ? "active" : ""}`}
      onClick={onClick}
      style={{ "--model-color": prediction.color }}
    >
      <div className="model-card-top">
        <div className="model-card-header">
          <div className="model-dot" style={{ background: prediction.color, boxShadow: `0 0 8px ${prediction.color}` }} />
          <div className="model-name">{prediction.display_name}</div>
          {rank !== null && (
            <div className="model-rank" style={{ color: rankColors[rank], borderColor: rankColors[rank] + "44" }}>
              {rankLabels[rank]}
            </div>
          )}
        </div>
        <div className="model-desc">{prediction.description}</div>
      </div>

      <div className="model-score-row">
        <div>
          <div className="model-score-label">Stability Score</div>
          <div className="model-score-val" style={{ color: prediction.stability.color }}>
            {Math.round(animated)}<span style={{ fontSize: "1rem", color: "var(--wm)" }}>/100</span>
          </div>
          <div className="model-status" style={{ color: prediction.stability.color }}>
            {prediction.stability.label}
          </div>
        </div>
        <div style={{ textAlign: "right" }}>
          <div className="model-score-label">Energy</div>
          <div className="model-energy-val">{prediction.energy_kcal}</div>
          <div className="model-score-label">kcal/mol</div>
        </div>
      </div>

      {/* Score bar */}
      <div className="model-bar-track">
        <div
          className="model-bar-fill"
          style={{
            width: `${animated}%`,
            background: prediction.stability.color,
            boxShadow: `0 0 8px ${prediction.stability.color}55`,
            transition: "width 1.4s cubic-bezier(.22,1,.36,1)",
          }}
        />
      </div>

      {/* Accuracy metrics */}
      {prediction.mae != null && (
        <div className="model-metrics">
          <div className="metric-chip">
            <span className="mc-label">MAE</span>
            <span className="mc-val" style={{ color: prediction.color }}>{prediction.mae}</span>
          </div>
          <div className="metric-chip">
            <span className="mc-label">R²</span>
            <span className="mc-val" style={{ color: prediction.color }}>{prediction.r2}</span>
          </div>
          <div className="metric-chip">
            <span className="mc-label">RMSE</span>
            <span className="mc-val" style={{ color: prediction.color }}>{prediction.rmse}</span>
          </div>
        </div>
      )}

      {!prediction.available && (
        <div className="model-unavailable">Model not trained yet — showing heuristic</div>
      )}
    </div>
  );
}

// ─── Accuracy Comparison Panel ────────────────────────────────────────────────
function AccuracyPanel({ predictions }) {
  const available = predictions.filter(p => p.mae != null);
  if (available.length === 0) {
    return (
      <div className="accuracy-empty">
        Train all models first to see accuracy comparison. Run: <code>python ml/train_model.py</code>
      </div>
    );
  }

  const bestMAE = Math.min(...available.map(p => p.mae));
  const maxMAE = Math.max(...available.map(p => p.mae));
  const bestR2 = Math.max(...available.map(p => p.r2));

  return (
    <div className="accuracy-panel">
      <div className="accuracy-section">
        <div className="acc-section-title">Mean Absolute Error (lower = better)</div>
        <div className="acc-bars">
          {predictions.map(p => (
            <div key={p.model_name} className="acc-bar-row">
              <div className="acc-bar-label" style={{ color: p.color }}>{p.display_name}</div>
              <div className="acc-bar-track">
                {p.mae != null ? (
                  <AccBar value={p.mae} max={maxMAE * 1.1} color={p.color} isBest={p.mae === bestMAE} />
                ) : (
                  <div className="acc-bar-na">Not trained</div>
                )}
              </div>
              <div className="acc-bar-num">{p.mae != null ? `${p.mae} kcal/mol` : "—"}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="accuracy-section">
        <div className="acc-section-title">R² Score (higher = better, max 1.0)</div>
        <div className="acc-bars">
          {predictions.map(p => (
            <div key={p.model_name} className="acc-bar-row">
              <div className="acc-bar-label" style={{ color: p.color }}>{p.display_name}</div>
              <div className="acc-bar-track">
                {p.r2 != null ? (
                  <AccBar value={p.r2} max={1.0} color={p.color} isBest={p.r2 === bestR2} />
                ) : (
                  <div className="acc-bar-na">Not trained</div>
                )}
              </div>
              <div className="acc-bar-num">{p.r2 != null ? p.r2 : "—"}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="accuracy-section">
        <div className="acc-section-title">Training Time</div>
        <div className="acc-table">
          <div className="acc-table-header">
            <span>Model</span><span>MAE</span><span>RMSE</span><span>R²</span><span>Train Time</span>
          </div>
          {predictions.map(p => (
            <div key={p.model_name} className="acc-table-row">
              <span style={{ color: p.color, fontWeight: 600 }}>{p.display_name}</span>
              <span>{p.mae != null ? `${p.mae}` : "—"}</span>
              <span>{p.rmse != null ? `${p.rmse}` : "—"}</span>
              <span className={p.r2 === bestR2 ? "best-val" : ""}>{p.r2 != null ? p.r2 : "—"}</span>
              <span>{p.train_time_s != null ? `${p.train_time_s}s` : "—"}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function AccBar({ value, max, color, isBest }) {
  const [w, setW] = useState(0);
  useEffect(() => {
    const t = setTimeout(() => setW((value / max) * 100), 200);
    return () => clearTimeout(t);
  }, [value, max]);

  return (
    <div style={{ position: "relative", height: "100%" }}>
      <div className="acc-fill" style={{
        width: `${w}%`,
        background: color,
        boxShadow: `0 0 8px ${color}44`,
        transition: "width 1.2s cubic-bezier(.22,1,.36,1)",
      }} />
      {isBest && <div className="best-badge" style={{ color }}>✓ best</div>}
    </div>
  );
}

// ─── Drug Row ─────────────────────────────────────────────────────────────────
function DrugRow({ match, index }) {
  const [w, setW] = useState(0);
  useEffect(() => {
    const t = setTimeout(() => setW(match.similarity), 200 + index * 100);
    return () => clearTimeout(t);
  }, [match.similarity, index]);
  const col = match.similarity >= 60 ? "var(--green)" : match.similarity >= 30 ? "var(--blue)" : "var(--red)";
  return (
    <div className="drug-entry">
      <div className="drow">
        <span className="dname">{match.name}</span>
        <span className="dpct" style={{ color: col }}>{match.similarity}%</span>
      </div>
      <div className="dtrack">
        <div className="dfill" style={{ width: `${w}%`, background: col, boxShadow: `0 0 8px ${col}44`, transition: `width 1.1s cubic-bezier(.22,1,.36,1) ${index * 0.1}s` }} />
      </div>
      <div className="dind">{match.indication}</div>
    </div>
  );
}

// ─── Lip Panel ────────────────────────────────────────────────────────────────
function LipinskiCard({ lipinski }) {
  return (
    <div className="lip-wrap">
      <div className="lip-top">
        <span className="lip-title">Lipinski Ro5</span>
        <span className={`lip-verdict ${lipinski.drug_like ? "pass" : "fail"}`}>{lipinski.verdict}</span>
      </div>
      {Object.entries(lipinski.rules).map(([rule, pass]) => (
        <div key={rule} className="lip-rule">
          <div className={`lip-dot ${pass ? "pass" : "fail"}`} />
          {rule}
        </div>
      ))}
    </div>
  );
}

// ─── 3D Viewer ────────────────────────────────────────────────────────────────
function MolViewer({ sdf, smiles }) {
  const ref = useRef(null);
  useEffect(() => {
    if (!window.$3Dmol || !ref.current) return;
    const viewer = window.$3Dmol.createViewer(ref.current, { backgroundColor: "0x030508" });
    try {
      if (sdf) viewer.addModel(sdf, "mol");
      else viewer.addModel(smiles, "smiles");
      viewer.setStyle({}, { stick: { radius: .13, colorscheme: "Jmol" }, sphere: { scale: .22, colorscheme: "Jmol" } });
      viewer.zoomTo(); viewer.spin("y", .5); viewer.render();
    } catch (e) {}
  }, [sdf, smiles]);
  return (
    <div ref={ref} style={{ width: "100%", height: "380px", borderRadius: "0", overflow: "hidden", position: "relative", zIndex: 1, display: "block" }} />
  );
}

// ─── Constants ────────────────────────────────────────────────────────────────
const EXAMPLES = [
  { label: "Ethanol", smiles: "CCO" },
  { label: "Aspirin", smiles: "CC(=O)Oc1ccccc1C(=O)O" },
  { label: "Caffeine", smiles: "Cn1c(=O)c2c(ncn2C)n(C)c1=O" },
  { label: "Dopamine", smiles: "NCCc1ccc(O)c(O)c1" },
  { label: "Benzene", smiles: "c1ccccc1" },
  { label: "Paracetamol", smiles: "CC(=O)Nc1ccc(O)cc1" },
];

const MSGS = [
  "Parsing molecular graph...",
  "Computing RDKit descriptors...",
  "Generating Morgan fingerprints...",
  "Running Gradient Boosting...",
  "Running XGBoost...",
  "Running Random Forest...",
  "Comparing model outputs...",
];

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const [smiles, setSmiles] = useState("");
  const [loading, setLoading] = useState(false);
  const [loadMsg, setLoadMsg] = useState("");
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [viz, setViz] = useState(null);
  const [tab, setTab] = useState("models");
  const [activeModel, setActiveModel] = useState(null);
  const [revealed, setRevealed] = useState(false);
  const [apiStatus, setApiStatus] = useState("checking");
  const [modelInfo, setModelInfo] = useState([]);

  useEffect(() => {
    fetch(`${API}/health`)
      .then(r => r.json())
      .then(d => setApiStatus(d.models_loaded?.length > 0 ? "online" : "partial"))
      .catch(() => setApiStatus("offline"));
    apiGet("/models").then(d => setModelInfo(d.models)).catch(() => {});
  }, []);

  const runLoadingSequence = useCallback(async () => {
    for (let i = 0; i < MSGS.length; i++) {
      setLoadMsg(MSGS[i]);
      await new Promise(r => setTimeout(r, 285));
    }
  }, []);

  const handleAnalyze = async (inputSmiles) => {
    const s = (inputSmiles || smiles).trim();
    if (!s) return;
    setLoading(true); setError(null); setResult(null);
    setViz(null); setRevealed(false); setTab("models"); setActiveModel(null);
    runLoadingSequence();

    try {
      const [res, vizRes] = await Promise.all([
        api("/predict_all", { smiles: s }),
        api("/visualize", { smiles: s }),
        new Promise(r => setTimeout(r, 2000)),
      ]);
      setResult(res);
      setViz(vizRes);
      setActiveModel(res.predictions[0]?.model_name || null);
      setTimeout(() => setRevealed(true), 80);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const activePred = result?.predictions?.find(p => p.model_name === activeModel);

  // Sort predictions by MAE for ranking (if available)
  const rankedPredictions = result?.predictions
    ? [...result.predictions].sort((a, b) => {
        if (a.mae == null && b.mae == null) return 0;
        if (a.mae == null) return 1;
        if (b.mae == null) return -1;
        return a.mae - b.mae;
      })
    : [];

  const getRank = (modelName) => {
    const i = rankedPredictions.findIndex(p => p.model_name === modelName);
    return result?.predictions?.every(p => p.mae == null) ? null : i;
  };

  return (
    <div className="root">
      <div className="orb o1" /><div className="orb o2" /><div className="orb o3" />
      {loading && <AtomLoader message={loadMsg} />}

      <div className="app">
        {/* Topbar */}
        <nav className="topbar">
          <div className="brand">
            <div className="brand-icon">Q</div>
            <div>
              <div className="brand-name">Q-Screen</div>
              <div className="brand-sub">Multi-Model Stability Platform</div>
            </div>
          </div>
          <div className="topbar-right">
            <div className="pill">v2.0</div>
            <div className={`pill ${apiStatus === "online" ? "on" : apiStatus === "partial" ? "warn" : ""}`}>
              ● {apiStatus === "online" ? "All Models Ready" : apiStatus === "partial" ? "Partial" : apiStatus === "offline" ? "Offline" : "Connecting"}
            </div>
          </div>
        </nav>

        <div className="main">
          {/* Input */}
          <div className="input-card">
            <div className="input-row">
              <div className="input-wrap">
                <input className="smiles-input" value={smiles}
                  onChange={e => setSmiles(e.target.value)}
                  onKeyDown={e => e.key === "Enter" && handleAnalyze()}
                  placeholder="Enter SMILES — e.g. CCO, CC(=O)Oc1ccccc1C(=O)O"
                  spellCheck={false} autoComplete="off" />
              </div>
              <button className="analyze-btn" onClick={() => handleAnalyze()} disabled={loading || !smiles}>
                Analyse →
              </button>
            </div>
            <div className="examples">
              <span className="ex-lbl">Try —</span>
              {EXAMPLES.map(ex => (
                <button key={ex.smiles} className="ex-chip"
                  onClick={() => { setSmiles(ex.smiles); handleAnalyze(ex.smiles); }}>
                  {ex.label}
                </button>
              ))}
            </div>
          </div>

          {error && <div className="err-card">{error}</div>}

          {result && (
            <div className={`results ${revealed ? "revealed" : ""}`}>

              {/* Formula strip */}
              <div className="formula-strip">
                <div className="formula-text">{result.descriptors.formula}</div>
                <div className="formula-smiles">{result.smiles}</div>
                <div className="formula-chips">
                  <span className="fchip fc-b">{result.descriptors.heavy_atoms} heavy atoms</span>
                  <span className="fchip fc-t">{result.descriptors.aromatic_rings} aromatic rings</span>
                  <span className="fchip fc-p">{result.descriptors.molecular_weight} Da</span>
                  <span className="fchip fc-dim">{result.processing_ms}ms</span>
                </div>
              </div>

              {/* Radial hero — driven by active model */}
              {activePred && (
                <div className="radial-wrap">
                  <div className="deco-ring dr1" />
                  <div className="deco-ring dr2" />
                  <div className="deco-ring dr3" />
                  <div className="sat-container">
                    <div className="sat sat-n">
                      <SatCard label="Ground-State Energy" value={activePred.energy_kcal} unit="kcal / mol" />
                    </div>
                    <div className="sat sat-e">
                      <SatCard label="Molecular Weight" value={result.descriptors.molecular_weight} unit="Daltons" sub={`LogP ${result.descriptors.logp}`} />
                    </div>
                    <div className="sat sat-s">
                      <SatCard label="Top Drug Match" value={result.similarity[0]?.name || "—"} unit={`${result.similarity[0]?.similarity || 0}% similarity`} />
                    </div>
                    <div className="sat sat-w">
                      <SatCard
                        label="Drug-Likeness"
                        value={result.lipinski.drug_like ? "Pass" : "Fail"}
                        unit={`${result.lipinski.violations} violation${result.lipinski.violations !== 1 ? "s" : ""}`}
                      />
                    </div>
                  </div>
                  <div className="center-orb">
                    <ScoreOrb
                      score={activePred.stability.score}
                      label={activePred.stability.label}
                      color={activePred.stability.color}
                      size={220}
                    />
                    <div className="orb-model-name" style={{ color: activePred.color }}>
                      {activePred.display_name}
                    </div>
                  </div>
                </div>
              )}

              {/* Tabs */}
              <div className="tab-strip">
                <button className={`tab-btn ${tab === "models" ? "active" : ""}`} onClick={() => setTab("models")}>
                  Models
                </button>
                <button className={`tab-btn ${tab === "compare" ? "active" : ""}`} onClick={() => setTab("compare")}>
                  Accuracy
                </button>
                <button className={`tab-btn ${tab === "properties" ? "active" : ""}`} onClick={() => setTab("properties")}>
                  Properties
                </button>
                <button className={`tab-btn ${tab === "drugs" ? "active" : ""}`} onClick={() => setTab("drugs")}>
                  Drug Match
                </button>
                <button className={`tab-btn ${tab === "structure" ? "active" : ""}`} onClick={() => setTab("structure")}>
                  3D Structure
                </button>
              </div>

              {/* Models tab — 3 cards side by side */}
              {tab === "models" && (
                <div className="tab-panel active">
                  <div className="model-cards-grid">
                    {result.predictions.map(pred => (
                      <ModelCard
                        key={pred.model_name}
                        prediction={pred}
                        isActive={activeModel === pred.model_name}
                        onClick={() => setActiveModel(pred.model_name)}
                        rank={getRank(pred.model_name)}
                      />
                    ))}
                  </div>
                  <div className="model-hint">
                    Click a model card to update the radial display above
                  </div>
                </div>
              )}

              {/* Accuracy comparison tab */}
              {tab === "compare" && (
                <div className="tab-panel active">
                  <AccuracyPanel predictions={result.predictions} />
                </div>
              )}

              {/* Properties tab */}
              {tab === "properties" && (
                <div className="tab-panel active">
                  <div className="detail-grid">
                    <div className="gpanel">
                      <div className="gph">
                        <div className="gpt"><div className="pi" style={{ background: "var(--blue)" }} />Descriptors</div>
                      </div>
                      <div className="gpb">
                        {[
                          { k: "Molecular Weight", v: `${result.descriptors.molecular_weight} Da`, w: result.descriptors.molecular_weight > 500 },
                          { k: "LogP", v: result.descriptors.logp, w: Math.abs(result.descriptors.logp) > 5 },
                          { k: "H-Bond Donors", v: result.descriptors.hbd, w: result.descriptors.hbd > 5 },
                          { k: "H-Bond Acceptors", v: result.descriptors.hba, w: result.descriptors.hba > 10 },
                          { k: "TPSA", v: `${result.descriptors.tpsa} Å²`, w: result.descriptors.tpsa > 140 },
                          { k: "Rotatable Bonds", v: result.descriptors.rotatable_bonds },
                          { k: "Aromatic Rings", v: result.descriptors.aromatic_rings },
                          { k: "Ring Count", v: result.descriptors.ring_count },
                          { k: "Frac CSP3", v: result.descriptors.frac_csp3 },
                          { k: "Heavy Atoms", v: result.descriptors.heavy_atoms },
                        ].map(row => (
                          <div key={row.k} className="desc-row">
                            <span className="dk">{row.k}</span>
                            <span className={`dv ${row.w ? "warn" : ""}`}>{row.v}</span>
                          </div>
                        ))}
                      </div>
                    </div>

                    <div className="gpanel">
                      <div className="gph">
                        <div className="gpt"><div className="pi" style={{ background: "var(--green)" }} />Lipinski Ro5</div>
                        <div className={`lip-verdict-badge ${result.lipinski.drug_like ? "pass" : "fail"}`}>
                          {result.lipinski.verdict}
                        </div>
                      </div>
                      <div className="gpb">
                        <LipinskiCard lipinski={result.lipinski} />
                      </div>
                    </div>

                    <div className="gpanel" style={{ background: "linear-gradient(145deg, rgba(79,158,255,0.08), rgba(45,212,191,0.04))" }}>
                      <div className="gph">
                        <div className="gpt"><div className="pi" style={{ background: "var(--teal)" }} />Energy Comparison</div>
                      </div>
                      <div className="gpb">
                        {result.predictions.map(p => (
                          <div key={p.model_name} className="energy-compare-row">
                            <div className="ec-model" style={{ color: p.color }}>{p.display_name}</div>
                            <div className="ec-energy">{p.energy_kcal}</div>
                            <div className="ec-score" style={{ color: p.stability.color }}>
                              {Math.round(p.stability.score)}/100
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Drug match tab */}
              {tab === "drugs" && (
                <div className="tab-panel active">
                  <div className="gpanel" style={{ marginBottom: 14 }}>
                    <div className="gph">
                      <div className="gpt"><div className="pi" style={{ background: "var(--purple)" }} />Tanimoto Structural Similarity</div>
                    </div>
                    <div className="gpb">
                      {result.similarity.map((m, i) => <DrugRow key={m.name} match={m} index={i} />)}
                    </div>
                  </div>
                </div>
              )}

              {/* 3D tab */}
              {tab === "structure" && (
                <div className="tab-panel active">
                  <div className="viewer-outer">
                    <MolViewer sdf={viz?.sdf} smiles={result.smiles} />
                    <div className="vfoot">
                      <span>{result.descriptors.formula} · {result.smiles}</span>
                      <span>Drag · Scroll · Right-click pan</span>
                    </div>
                  </div>
                </div>
              )}

            </div>
          )}
        </div>

        <footer className="footer">
          <span><span className="fb">Q-Screen</span> · Multi-Model Molecular Stability Platform</span>
          <span>Research only · Not clinically validated · GradientBoosting + XGBoost + RandomForest</span>
        </footer>
      </div>
    </div>
  );
}
