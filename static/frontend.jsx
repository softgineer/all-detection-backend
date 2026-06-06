import { useState, useRef, useCallback, useEffect } from "react";

// ── CONFIG: point this at your deployed backend ──────────────────────────
const API_BASE = "http://localhost:8000"; // change after deploying

// ── Styles injected once ──────────────────────────────────────────────────
const CSS = `
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@400;600;700;800&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:        #060f1a;
    --surface:   #0d1e30;
    --surface2:  #112238;
    --border:    #1e3a5a;
    --accent:    #00e5ff;
    --accent2:   #ff3d71;
    --positive:  #00e676;
    --negative:  #ff3d71;
    --text:      #e8f4fd;
    --muted:     #5a8ab0;
    --mono:      'Space Mono', monospace;
    --sans:      'Syne', sans-serif;
  }

  body { background: var(--bg); color: var(--text); font-family: var(--sans); min-height: 100vh; overflow-x: hidden; }

  .scanline {
    position: fixed; inset: 0; pointer-events: none; z-index: 9999;
    background: repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,0,0,0.03) 2px, rgba(0,0,0,0.03) 4px);
  }

  .grid-bg {
    position: fixed; inset: 0; pointer-events: none; z-index: 0;
    background-image: linear-gradient(rgba(0,229,255,0.03) 1px, transparent 1px),
                      linear-gradient(90deg, rgba(0,229,255,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
  }

  .app { position: relative; z-index: 1; min-height: 100vh; display: flex; flex-direction: column; }

  /* HEADER */
  .header {
    border-bottom: 1px solid var(--border);
    padding: 20px 40px;
    display: flex; align-items: center; justify-content: space-between;
    background: rgba(6,15,26,0.9); backdrop-filter: blur(12px);
    position: sticky; top: 0; z-index: 100;
  }
  .header-logo { display: flex; align-items: center; gap: 12px; }
  .logo-mark {
    width: 36px; height: 36px; border: 2px solid var(--accent);
    border-radius: 8px; display: flex; align-items: center; justify-content: center;
    font-family: var(--mono); font-size: 14px; font-weight: 700; color: var(--accent);
    position: relative; overflow: hidden;
  }
  .logo-mark::after {
    content: ''; position: absolute; inset: 0;
    background: linear-gradient(135deg, rgba(0,229,255,0.15), transparent);
  }
  .header-title { font-size: 15px; font-weight: 700; letter-spacing: 0.08em; color: var(--text); }
  .header-sub   { font-size: 11px; color: var(--muted); font-family: var(--mono); margin-top: 2px; }
  .status-pill  {
    display: flex; align-items: center; gap: 6px;
    padding: 6px 14px; border-radius: 20px;
    border: 1px solid var(--border); font-family: var(--mono); font-size: 11px; color: var(--muted);
  }
  .status-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--positive); animation: pulse 2s infinite; }
  @keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.4; } }

  /* MAIN */
  .main { flex: 1; padding: 48px 40px; max-width: 1200px; margin: 0 auto; width: 100%; }

  /* HERO */
  .hero { text-align: center; margin-bottom: 56px; }
  .hero-tag {
    display: inline-block; font-family: var(--mono); font-size: 11px; letter-spacing: 0.12em;
    color: var(--accent); border: 1px solid rgba(0,229,255,0.3);
    padding: 5px 14px; border-radius: 4px; margin-bottom: 24px;
    background: rgba(0,229,255,0.05);
  }
  .hero h1 { font-size: clamp(28px,5vw,52px); font-weight: 800; line-height: 1.1; letter-spacing: -0.02em; margin-bottom: 16px; }
  .hero h1 span { color: var(--accent); }
  .hero p { font-size: 15px; color: var(--muted); max-width: 520px; margin: 0 auto; line-height: 1.7; }

  /* UPLOAD ZONE */
  .upload-zone {
    border: 2px dashed var(--border); border-radius: 16px;
    padding: 56px 40px; text-align: center; cursor: pointer;
    transition: all 0.25s; background: var(--surface);
    position: relative; overflow: hidden;
  }
  .upload-zone:hover, .upload-zone.drag-over {
    border-color: var(--accent); background: rgba(0,229,255,0.04);
    transform: translateY(-2px); box-shadow: 0 20px 60px rgba(0,229,255,0.08);
  }
  .upload-zone::before {
    content: ''; position: absolute; inset: 0;
    background: radial-gradient(ellipse at center, rgba(0,229,255,0.04) 0%, transparent 70%);
    opacity: 0; transition: opacity 0.3s;
  }
  .upload-zone:hover::before, .upload-zone.drag-over::before { opacity: 1; }
  .upload-icon {
    width: 64px; height: 64px; margin: 0 auto 20px;
    border: 2px solid var(--border); border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 28px; transition: all 0.25s;
  }
  .upload-zone:hover .upload-icon { border-color: var(--accent); transform: scale(1.05); }
  .upload-title { font-size: 18px; font-weight: 700; margin-bottom: 8px; }
  .upload-desc  { font-size: 13px; color: var(--muted); font-family: var(--mono); }

  /* PREVIEW */
  .preview-area { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 20px; }
  .preview-card { border-radius: 12px; overflow: hidden; border: 1px solid var(--border); background: var(--surface); }
  .preview-card img { width: 100%; height: 200px; object-fit: cover; display: block; }
  .preview-card-label { padding: 10px 14px; font-family: var(--mono); font-size: 11px; color: var(--muted); border-top: 1px solid var(--border); }

  /* ANALYSE BUTTON */
  .btn-analyse {
    width: 100%; padding: 16px; border-radius: 10px; border: none; cursor: pointer;
    background: linear-gradient(135deg, var(--accent), #0091ea);
    color: #060f1a; font-family: var(--sans); font-size: 15px; font-weight: 700;
    letter-spacing: 0.04em; transition: all 0.2s; margin-top: 4px;
    display: flex; align-items: center; justify-content: center; gap: 8px;
  }
  .btn-analyse:hover:not(:disabled) { transform: translateY(-2px); box-shadow: 0 12px 40px rgba(0,229,255,0.3); }
  .btn-analyse:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }

  /* RESULTS */
  .results-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 32px; }

  .result-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 16px; padding: 24px;
  }
  .result-card.verdict { grid-column: 1/-1; }

  .verdict-inner { display: flex; align-items: center; gap: 24px; flex-wrap: wrap; }
  .verdict-badge {
    padding: 12px 28px; border-radius: 10px; font-size: 22px; font-weight: 800;
    letter-spacing: 0.04em; white-space: nowrap;
  }
  .verdict-badge.positive { background: rgba(255,61,113,0.15); color: var(--negative); border: 1px solid rgba(255,61,113,0.3); }
  .verdict-badge.negative { background: rgba(0,230,118,0.12); color: var(--positive); border: 1px solid rgba(0,230,118,0.3); }

  .confidence-bar { flex: 1; min-width: 200px; }
  .conf-label { display: flex; justify-content: space-between; font-family: var(--mono); font-size: 12px; color: var(--muted); margin-bottom: 8px; }
  .conf-track { height: 8px; background: var(--surface2); border-radius: 4px; overflow: hidden; }
  .conf-fill  { height: 100%; border-radius: 4px; transition: width 1s cubic-bezier(.4,0,.2,1); }
  .conf-fill.positive { background: linear-gradient(90deg, #ff3d71, #ff6d94); }
  .conf-fill.negative { background: linear-gradient(90deg, #00e676, #69f0ae); }

  .card-title { font-size: 11px; font-family: var(--mono); letter-spacing: 0.1em; color: var(--muted); margin-bottom: 16px; text-transform: uppercase; }

  /* Votes */
  .votes-row { display: flex; gap: 12px; }
  .vote-chip {
    flex: 1; padding: 14px 10px; border-radius: 10px;
    border: 1px solid var(--border); text-align: center;
    font-family: var(--mono); font-size: 12px;
  }
  .vote-chip.voted-yes { border-color: rgba(255,61,113,0.4); background: rgba(255,61,113,0.08); }
  .vote-chip.voted-no  { border-color: rgba(0,230,118,0.3);  background: rgba(0,230,118,0.06); }
  .vote-chip-name { color: var(--muted); font-size: 10px; display: block; margin-bottom: 6px; }
  .vote-chip-val  { font-size: 11px; font-weight: 700; }
  .vote-chip-prob { font-size: 10px; color: var(--muted); margin-top: 3px; display: block; }

  /* Probability bars */
  .prob-row { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
  .prob-label { font-family: var(--mono); font-size: 11px; color: var(--muted); width: 60px; flex-shrink: 0; }
  .prob-bar-track { flex: 1; height: 6px; background: var(--surface2); border-radius: 3px; overflow: hidden; }
  .prob-bar-fill  { height: 100%; border-radius: 3px; }
  .prob-pct { font-family: var(--mono); font-size: 11px; width: 40px; text-align: right; flex-shrink: 0; }

  /* Feature grid */
  .feature-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
  .feat-item { background: var(--surface2); border-radius: 8px; padding: 10px 12px; }
  .feat-key  { font-family: var(--mono); font-size: 10px; color: var(--muted); margin-bottom: 4px; }
  .feat-val  { font-family: var(--mono); font-size: 13px; font-weight: 700; color: var(--text); }

  /* Preview segmented */
  .seg-img { width: 100%; border-radius: 10px; border: 1px solid var(--border); display: block; }

  /* Loading spinner */
  .spinner { width: 20px; height: 20px; border: 2px solid rgba(6,15,26,0.3); border-top-color: #060f1a; border-radius: 50%; animation: spin 0.7s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* Error */
  .error-box { background: rgba(255,61,113,0.1); border: 1px solid rgba(255,61,113,0.3); border-radius: 10px; padding: 16px 20px; font-family: var(--mono); font-size: 13px; color: var(--negative); margin-top: 16px; }

  /* Footer */
  .footer { border-top: 1px solid var(--border); padding: 24px 40px; text-align: center; font-family: var(--mono); font-size: 11px; color: var(--muted); }

  @media (max-width: 700px) {
    .main { padding: 24px 16px; }
    .header { padding: 16px; }
    .results-grid { grid-template-columns: 1fr; }
    .preview-area { grid-template-columns: 1fr; }
    .votes-row { flex-direction: column; }
    .feature-grid { grid-template-columns: 1fr; }
  }
`;

// ── Inject styles ──────────────────────────────────────────────────────────
function StyleTag() {
  useEffect(() => {
    const el = document.createElement("style");
    el.textContent = CSS;
    document.head.appendChild(el);
    return () => document.head.removeChild(el);
  }, []);
  return null;
}

// ── Sub-components ─────────────────────────────────────────────────────────
function ProbBar({ label, pct, color }) {
  return (
    <div className="prob-row">
      <span className="prob-label">{label}</span>
      <div className="prob-bar-track">
        <div className="prob-bar-fill"
          style={{ width: `${pct}%`, background: color, transition: "width 1s cubic-bezier(.4,0,.2,1)" }} />
      </div>
      <span className="prob-pct" style={{ color }}>{pct}%</span>
    </div>
  );
}

function VoteChip({ name, vote, prob }) {
  const yes = vote === 1;
  return (
    <div className={`vote-chip ${yes ? "voted-yes" : "voted-no"}`}>
      <span className="vote-chip-name">{name}</span>
      <span className="vote-chip-val" style={{ color: yes ? "var(--negative)" : "var(--positive)" }}>
        {yes ? "ALL +" : "Normal"}
      </span>
      <span className="vote-chip-prob">{prob}%</span>
    </div>
  );
}

function FeatureCard({ features }) {
  const labels = {
    nucleus_area_px:  "Nucleus Area (px)",
    nucleus_pixels:   "Nucleus Pixels",
    circularity:      "Circularity",
    solidity:         "Solidity",
    nc_ratio:         "N/C Ratio",
    eccentricity:     "Eccentricity",
    glcm_contrast:    "GLCM Contrast",
    glcm_energy:      "GLCM Energy",
    glcm_homogeneity: "GLCM Homogeneity",
    glcm_correlation: "GLCM Correlation",
    mean_intensity_R: "Mean R Channel",
    mean_intensity_G: "Mean G Channel",
    mean_intensity_B: "Mean B Channel",
  };
  return (
    <div className="feature-grid">
      {Object.entries(features).map(([k, v]) => (
        <div key={k} className="feat-item">
          <div className="feat-key">{labels[k] || k}</div>
          <div className="feat-val">{typeof v === "number" ? v.toFixed(typeof v === "number" && v > 100 ? 0 : 4) : v}</div>
        </div>
      ))}
    </div>
  );
}

// ── Main App ───────────────────────────────────────────────────────────────
export default function App() {
  const [file,     setFile]     = useState(null);
  const [preview,  setPreview]  = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const [loading,  setLoading]  = useState(false);
  const [result,   setResult]   = useState(null);
  const [error,    setError]    = useState(null);
  const inputRef = useRef();

  const handleFile = useCallback((f) => {
    if (!f) return;
    setFile(f);
    setResult(null);
    setError(null);
    const url = URL.createObjectURL(f);
    setPreview(url);
  }, []);

  const onDrop = useCallback((e) => {
    e.preventDefault(); setDragOver(false);
    const f = e.dataTransfer.files[0];
    if (f && f.type.startsWith("image/")) handleFile(f);
  }, [handleFile]);

  const analyse = async () => {
    if (!file) return;
    setLoading(true); setResult(null); setError(null);
    const form = new FormData();
    form.append("file", file);
    try {
      const res = await fetch(`${API_BASE}/api/predict`, { method: "POST", body: form });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: "Unknown error" }));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      const data = await res.json();
      setResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const isPos = result?.label_id === 1;

  return (
    <>
      <StyleTag />
      <div className="scanline" />
      <div className="grid-bg" />
      <div className="app">

        {/* HEADER */}
        <header className="header">
          <div className="header-logo">
            <div className="logo-mark">ALL</div>
            <div>
              <div className="header-title">ALL Detection System</div>
              <div className="header-sub">Multi-Stage Ensemble Learning · FUTA BTech 2025</div>
            </div>
          </div>
          <div className="status-pill">
            <div className="status-dot" />
            System Online
          </div>
        </header>

        {/* MAIN */}
        <main className="main">

          {/* HERO */}
          <div className="hero">
            <div className="hero-tag">/ LEUKEMIA DETECTION ENGINE /</div>
            <h1>Detect <span>Acute Lymphoblastic</span><br />Leukemia Instantly</h1>
            <p>Upload a peripheral blood smear image. The ensemble of SVM, Random Forest, and Gradient Boosting classifiers will analyse it in seconds.</p>
          </div>

          {/* UPLOAD */}
          <div
            className={`upload-zone ${dragOver ? "drag-over" : ""}`}
            onClick={() => inputRef.current?.click()}
            onDragOver={e => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={onDrop}
          >
            <input ref={inputRef} type="file" accept="image/*" style={{ display: "none" }}
              onChange={e => handleFile(e.target.files[0])} />
            {!file ? (
              <>
                <div className="upload-icon">🔬</div>
                <div className="upload-title">Drop blood smear image here</div>
                <div className="upload-desc">PNG · JPG · BMP · TIFF · max 20 MB</div>
              </>
            ) : (
              <>
                <div className="preview-area" onClick={e => e.stopPropagation()}>
                  <div className="preview-card">
                    <img src={preview} alt="Uploaded" />
                    <div className="preview-card-label">📁 Original · {file.name}</div>
                  </div>
                  {result?.preview_b64 && (
                    <div className="preview-card">
                      <img src={`data:image/png;base64,${result.preview_b64}`} alt="Processed" />
                      <div className="preview-card-label">🧬 Processed · Nucleus segmented</div>
                    </div>
                  )}
                </div>
                <div style={{ color: "var(--muted)", fontSize: 12, fontFamily: "var(--mono)", marginTop: 4 }}>
                  Click to change image
                </div>
              </>
            )}
          </div>

          {/* ANALYSE BUTTON */}
          {file && (
            <button className="btn-analyse" onClick={analyse} disabled={loading}>
              {loading ? <><div className="spinner" /> Analysing…</> : "⚡ Run Ensemble Analysis"}
            </button>
          )}

          {/* ERROR */}
          {error && (
            <div className="error-box">
              ⚠ {error}<br />
              <span style={{ opacity: 0.7, fontSize: 11 }}>
                Make sure the backend is running at: {API_BASE}
              </span>
            </div>
          )}

          {/* RESULTS */}
          {result && (
            <div className="results-grid">

              {/* Verdict */}
              <div className="result-card verdict">
                <div className="card-title">/ Classification Result /</div>
                <div className="verdict-inner">
                  <div className={`verdict-badge ${isPos ? "positive" : "negative"}`}>
                    {result.label}
                  </div>
                  <div className="confidence-bar">
                    <div className="conf-label">
                      <span>Confidence</span>
                      <span style={{ color: isPos ? "var(--negative)" : "var(--positive)" }}>
                        {result.confidence}%
                      </span>
                    </div>
                    <div className="conf-track">
                      <div className={`conf-fill ${isPos ? "positive" : "negative"}`}
                        style={{ width: `${result.confidence}%` }} />
                    </div>
                  </div>
                  <div style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--muted)" }}>
                    ⏱ {result.inference_ms} ms
                  </div>
                </div>
              </div>

              {/* Classifier Votes */}
              <div className="result-card">
                <div className="card-title">/ Classifier Votes /</div>
                <div className="votes-row">
                  <VoteChip name="SVM" vote={result.votes.SVM} prob={result.classifier_probs.SVM} />
                  <VoteChip name="Random Forest" vote={result.votes.RF} prob={result.classifier_probs.RF} />
                  <VoteChip name="Grad. Boost" vote={result.votes.GB} prob={result.classifier_probs.GB} />
                </div>
              </div>

              {/* Probabilities */}
              <div className="result-card">
                <div className="card-title">/ Class Probabilities /</div>
                <ProbBar label="ALL +" pct={result.probabilities.ALL}
                  color="var(--negative)" />
                <ProbBar label="Normal" pct={result.probabilities.Normal}
                  color="var(--positive)" />
                <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--muted)", marginTop: 12 }}>
                  Soft vote: average of SVM + RF + GB probabilities
                </div>
              </div>

              {/* Diagnostic Features */}
              <div className="result-card" style={{ gridColumn: "1/-1" }}>
                <div className="card-title">/ Extracted Diagnostic Features /</div>
                <FeatureCard features={result.features} />
              </div>

            </div>
          )}
        </main>

        <footer className="footer">
          Multi-Stage Ensemble Learning System · Goodness Ikubuwaje Oluwasegun · SEN/20/5102<br />
          FUTA — Dept. of Software Engineering · Supervisor: Dr. Mrs. O. V. Olatunde
        </footer>
      </div>
    </>
  );
}
