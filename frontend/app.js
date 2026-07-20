const API_BASE = ""; // same-origin — FastAPI serves this frontend directly

const els = {
  question:      document.getElementById("question"),
  nSamples:      document.getElementById("nSamples"),
  nHint:         document.getElementById("nHint"),
  temperature:   document.getElementById("temperature"),
  runBtn:        document.getElementById("runBtn"),
  btnText:       document.querySelector("#runBtn") ,
  results:       document.getElementById("results"),
  errorPanel:    document.getElementById("errorPanel"),
  errorText:     document.getElementById("errorText"),
  statusDot:     document.getElementById("statusDot"),
  statusText:    document.getElementById("statusText"),
  gaugeArc:      document.getElementById("gaugeArc"),
  gaugeNeedle:   document.getElementById("gaugeNeedle"),
  riskNumber:    document.getElementById("riskNumber"),
  verdict:       document.getElementById("verdict"),
  meanSim:       document.getElementById("meanSim"),
  meanSimBar:    document.getElementById("meanSimBar"),
  divergence:    document.getElementById("divergence"),
  divergenceBar: document.getElementById("divergenceBar"),
  entropy:       document.getElementById("entropy"),
  providerModel: document.getElementById("providerModel"),
  explanation:   document.getElementById("explanation"),
  matrixGrid:    document.getElementById("matrixGrid"),
  samplesList:   document.getElementById("samplesList"),
};

// Keep hint in sync
els.nSamples.addEventListener("input", () => {
  els.nHint.textContent = els.nSamples.value;
});

// ── Health check ──────────────────────────────────────────────
async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    if (!res.ok) throw new Error("bad status");
    const data = await res.json();
    els.statusDot.classList.add("ok");
    els.statusText.textContent = `${data.provider} · ${data.model}`;
  } catch (e) {
    els.statusDot.classList.add("err");
    els.statusText.textContent = "backend unreachable";
  }
}

// ── Color helpers ─────────────────────────────────────────────
function riskColor(score) {
  if (score < 0.30) return "var(--green)";
  if (score < 0.60) return "var(--amber)";
  return "var(--red)";
}

function verdictClass(verdict) {
  if (verdict === "likely grounded")     return "grounded";
  if (verdict === "likely hallucinated") return "hallucinated";
  return "uncertain";
}

// ── Gauge ─────────────────────────────────────────────────────
function renderGauge(score) {
  const arcLen = 333; // semicircle arc r=106
  const filled = score * arcLen;

  els.gaugeArc.setAttribute("stroke-dasharray", `${filled} ${arcLen}`);
  els.gaugeArc.style.stroke = riskColor(score);
  if (els.gaugeGlow) {
    els.gaugeGlow.setAttribute("stroke-dasharray", `${filled} ${arcLen}`);
  }

  // Needle: -90deg (left, 0) → +90deg (right, 1)
  const angle = -90 + score * 180;
  els.gaugeNeedle.style.transform = `rotate(${angle}deg)`;

  els.riskNumber.textContent = score.toFixed(2);
  els.riskNumber.style.color = riskColor(score);
}

// ── Matrix ────────────────────────────────────────────────────
function cellColor(value) {
  // 1 = identical (dark teal), 0 = completely different (light/transparent)
  const alpha = 0.1 + value * 0.75;
  const hue = 220 + value * 30; // shift from blue to teal as similarity grows
  return `hsla(${hue}, 60%, ${20 + value * 30}%, ${alpha})`;
}

function renderMatrix(matrix) {
  const n = matrix.length;
  els.matrixGrid.style.gridTemplateColumns = `repeat(${n}, 38px)`;
  els.matrixGrid.innerHTML = "";
  matrix.forEach((row) => {
    row.forEach((value) => {
      const cell = document.createElement("div");
      cell.className = "matrix-cell";
      cell.style.background = cellColor(value);
      cell.textContent = value.toFixed(2);
      cell.title = `Similarity: ${(value * 100).toFixed(1)}%`;
      els.matrixGrid.appendChild(cell);
    });
  });
}

// ── Samples ───────────────────────────────────────────────────
function renderSamples(samples) {
  els.samplesList.innerHTML = "";
  samples.forEach((s, i) => {
    const div = document.createElement("div");
    div.className = "sample-item";
    div.style.animationDelay = `${i * 60}ms`;
    div.innerHTML = `<span class="sample-index">#${s.index + 1}</span>${escapeHtml(s.text)}`;
    els.samplesList.appendChild(div);
  });
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

// ── Animate number counting up ────────────────────────────────
function animateNumber(el, target, decimals = 3, duration = 700) {
  const start = performance.now();
  function step(now) {
    const t = Math.min((now - start) / duration, 1);
    const ease = 1 - Math.pow(1 - t, 3);
    el.textContent = (target * ease).toFixed(decimals);
    if (t < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

// ── Main analysis ─────────────────────────────────────────────
async function runAnalysis() {
  const question = els.question.value.trim();
  if (!question) {
    els.question.focus();
    els.question.style.borderColor = "rgba(248,113,113,0.5)";
    setTimeout(() => { els.question.style.borderColor = ""; }, 1500);
    return;
  }

  els.runBtn.disabled = true;
  els.runBtn.querySelector(".btn-icon") && (els.runBtn.querySelector("svg").style.display = "none");
  const origText = els.runBtn.childNodes[0];
  if (origText.nodeType === 3) origText.textContent = "Running…";
  els.errorPanel.hidden = true;
  els.results.hidden = true;

  try {
    const res = await fetch(`${API_BASE}/api/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        n_samples:   parseInt(els.nSamples.value, 10),
        temperature: parseFloat(els.temperature.value),
      }),
    });

    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Request failed");

    // Gauge + verdict
    renderGauge(data.risk_score);
    els.verdict.textContent = data.verdict.toUpperCase();
    els.verdict.className = `verdict-badge ${verdictClass(data.verdict)}`;

    // Metric values with animation
    animateNumber(els.meanSim,   data.mean_pairwise_similarity, 3);
    animateNumber(els.divergence, data.semantic_divergence, 3);

    // Metric bars
    setTimeout(() => {
      els.meanSimBar.style.width    = `${data.mean_pairwise_similarity * 100}%`;
      els.divergenceBar.style.width = `${data.semantic_divergence * 100}%`;
    }, 100);

    els.entropy.textContent = data.entropy_score !== null
      ? data.entropy_score.toFixed(3)
      : "n/a";

    els.providerModel.textContent = `${data.provider} / ${data.model}`;
    els.explanation.textContent   = data.explanation;

    renderMatrix(data.pairwise_similarity);
    renderSamples(data.samples);

    els.results.hidden = false;
    els.results.scrollIntoView({ behavior: "smooth", block: "start" });

  } catch (e) {
    els.errorText.textContent = `Error: ${e.message}`;
    els.errorPanel.hidden = false;
  } finally {
    els.runBtn.disabled = false;
    els.btnText.textContent = "Run Analysis";
    els.btnText.classList.remove("running-pulse");
  }
}

els.runBtn.addEventListener("click", runAnalysis);

// Ctrl+Enter shortcut
document.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") runAnalysis();
});

checkHealth();
