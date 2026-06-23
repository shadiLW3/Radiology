// MedVS-AI Phase 1 — vanilla JS + Konva. Draw the lesion -> lock -> reveal vs the model.
const DISPLAY = 480;        // stage size in px (drawing happens 1:1 in image space)
const GRID = 256;           // server scoring grid; export at this resolution

// --- session ---
let sessionId = localStorage.getItem("medvs_session");
if (!sessionId) { sessionId = "s_" + Math.random().toString(36).slice(2) + Date.now().toString(36); localStorage.setItem("medvs_session", sessionId); }

// --- state ---
const state = { tool: "lasso", brush: 22, dx: null, conf: 50, locked: false, caseId: null, t0: 0, shapes: [], drawing: null, expertBand: [0.75, 0.81], verifiedBadge: null };

// --- Konva ---
const stage = new Konva.Stage({ container: "stage", width: DISPLAY, height: DISPLAY });
const imgLayer = new Konva.Layer({ listening: false });
const drawLayer = new Konva.Layer();
stage.add(imgLayer); stage.add(drawLayer);

// --- helpers ---
const $ = (id) => document.getElementById(id);
function setTool(t) {
  state.tool = t;
  ["lasso", "brush", "eraser"].forEach((k) => $("tool-" + k).classList.toggle("active", k === t));
}
function updateLockEnabled() { $("lock").disabled = !(state.dx && state.shapes.length > 0 && !state.locked); }

function loadCaseImage(url) {
  return new Promise((res) => {
    const im = new Image();
    im.crossOrigin = "anonymous";
    im.onload = () => {
      imgLayer.destroyChildren();
      imgLayer.add(new Konva.Image({ image: im, width: DISPLAY, height: DISPLAY }));
      imgLayer.draw(); res();
    };
    im.src = url + "?t=" + Date.now();
  });
}

async function loadNextCase() {
  const r = await fetch(`/api/next_case?session_id=${encodeURIComponent(sessionId)}`);
  const data = await r.json();
  $("reveal-card").classList.remove("show");
  state.locked = false; state.dx = null; state.shapes = []; state.drawing = null;
  drawLayer.destroyChildren(); drawLayer.draw();
  document.querySelectorAll(".dx").forEach((b) => b.classList.remove("active"));
  if (!data.case_id) { $("case-title").textContent = "🎉 You've seen every case — nice work."; $("lock").disabled = true; return; }
  state.caseId = data.case_id;
  state.expertBand = data.expert_dice_band || state.expertBand;
  $("case-title").textContent = "Case " + data.case_id + " — trace the lesion border";
  await loadCaseImage(data.image_url);
  state.t0 = Date.now();
  updateLockEnabled();
}

// --- drawing ---
function strokeOpts() {
  if (state.tool === "eraser")
    return { stroke: "#000", strokeWidth: state.brush, lineCap: "round", lineJoin: "round", globalCompositeOperation: "destination-out" };
  if (state.tool === "brush")
    return { stroke: "rgba(229,57,53,0.55)", strokeWidth: state.brush, lineCap: "round", lineJoin: "round" };
  return { stroke: "rgba(229,57,53,0.9)", strokeWidth: 2, closed: false, fill: "rgba(229,57,53,0.4)" }; // lasso
}
stage.on("pointerdown", () => {
  if (state.locked) return;
  const p = stage.getPointerPosition();
  const line = new Konva.Line({ points: [p.x, p.y], ...strokeOpts() });
  drawLayer.add(line); state.drawing = line;
});
stage.on("pointermove", () => {
  if (!state.drawing) return;
  const p = stage.getPointerPosition();
  state.drawing.points(state.drawing.points().concat([p.x, p.y]));
  drawLayer.batchDraw();
});
stage.on("pointerup", () => {
  if (!state.drawing) return;
  if (state.tool === "lasso") state.drawing.closed(true);
  drawLayer.batchDraw();
  state.shapes.push(state.drawing); state.drawing = null;
  updateLockEnabled();
});

function exportMaskDataUrl() { return drawLayer.toDataURL({ pixelRatio: GRID / DISPLAY, mimeType: "image/png" }); }

// --- submit / reveal ---
async function submit() {
  state.locked = true; $("lock").disabled = true;
  const body = {
    session_id: sessionId, case_id: state.caseId, badge: $("badge").value,
    diagnosis: state.dx, confidence: state.conf, mask_png: exportMaskDataUrl(),
    draw_ms: Date.now() - state.t0,
  };
  const r = await fetch("/api/attempt", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const d = await r.json();
  renderReveal(d, body.mask_png);
  loadLeaderboard();
}

function fmt(v) { return v === null || v === undefined ? "—" : v; }
function renderReveal(d, youMask) {
  $("rv-you").src = youMask;
  $("rv-model").src = d.masks.model_url + "?t=" + Date.now();
  $("rv-gt").src = d.masks.gt_url + "?t=" + Date.now();
  const v = $("verdict");
  v.textContent = d.beat_model_on_dice ? "🏆 You beat the model on Dice!" : "The model edged you out on Dice.";
  v.className = "verdict " + (d.beat_model_on_dice ? "win" : "lose");
  const pill = (ok) => ok ? '<span class="pill ok">correct</span>' : '<span class="pill no">incorrect</span>';
  const modelDx = d.model.diagnosis ? `${d.model.diagnosis} ${pill(d.model.diagnosis_correct)}` : "abstains (seg-only)";
  let rows = `
    <tr><th>Metric</th><th>You</th><th>Model</th></tr>
    <tr><td>Dice ↑</td><td>${fmt(d.you.dice)}</td><td>${fmt(d.model.dice)}</td></tr>
    <tr><td>IoU ↑</td><td>${fmt(d.you.iou)}</td><td>${fmt(d.model.iou)}</td></tr>
    <tr><td>Hausdorff95 ↓ (px)</td><td>${fmt(d.you.hausdorff95)}</td><td>${fmt(d.model.hausdorff95)}</td></tr>
    <tr><td>Diagnosis</td><td>${d.you.diagnosis} ${pill(d.you.diagnosis_correct)}</td><td>${modelDx}</td></tr>
    <tr><td>Ground truth dx</td><td colspan="2">${d.ground_truth.diagnosis}</td></tr>`;
  if (d.agreed_with_model !== null && d.agreed_with_model !== undefined)
    rows += `<tr><td>You &amp; model agree?</td><td colspan="2">${d.agreed_with_model ? "yes" : "no"}</td></tr>`;
  $("metrics-table").innerHTML = rows;
  const lo = (d.expert_dice_band || state.expertBand)[0], hi = (d.expert_dice_band || state.expertBand)[1];
  let note = `For context, two human experts typically agree only ~${lo}–${hi} Dice with each other on lesion borders — so don't expect 0.9+.`;
  if (!d.model.diagnosis) note += " The model is segmentation-only and abstains on diagnosis; a benign/malignant classifier is a planned next step.";
  $("context-note").textContent = note;
  $("reveal-card").classList.add("show");
}

async function loadLeaderboard() {
  const d = await (await fetch("/api/leaderboard")).json();
  let rows = `<tr><th>Background</th><th>n</th><th>Avg Dice</th><th>Dx accuracy</th><th>Beat-model rate</th></tr>`;
  for (const b of d.by_badge) {
    const tag = b.verified ? ' <span class="pill ok">✓ verified</span>' : "";
    rows += `<tr><td>${b.badge}${tag}</td><td>${b.n}</td><td>${b.avg_dice}</td><td>${(b.diagnosis_accuracy * 100).toFixed(0)}%</td><td>${(b.beat_model_rate * 100).toFixed(0)}%</td></tr>`;
  }
  if (d.model_avg_dice !== null)
    rows += `<tr class="model-row"><td>★ Model</td><td>—</td><td>${d.model_avg_dice}</td><td>—</td><td>—</td></tr>`;
  $("leaderboard").innerHTML = rows;
}

// --- NPI credential verification ---
function renderCredential(d) {
  const el = $("cred-status");
  if (d.badge) {
    state.verifiedBadge = d.badge;
    const nm = d.name_match === false ? " (name didn't match — counted as unverified-name)" : "";
    el.innerHTML = `<span class="pill ok">✓ Verified: ${d.badge}${d.specialty ? " — " + d.specialty : ""}</span> Your attempts now count as <b>${d.badge}</b>.${nm}`;
    $("badge").disabled = true;
  } else {
    state.verifiedBadge = null;
    el.textContent = "Not verified — you'll play as your self-reported background.";
    $("badge").disabled = false;
  }
}
async function loadCredential() {
  const d = await (await fetch(`/api/credential?session_id=${encodeURIComponent(sessionId)}`)).json();
  renderCredential(d);
}
async function verifyNpi() {
  const body = { session_id: sessionId, npi: $("npi").value.trim(), first_name: $("npi-first").value.trim(), last_name: $("npi-last").value.trim() };
  $("verify-btn").disabled = true; $("verify-btn").textContent = "Checking…";
  const d = await (await fetch("/api/verify_npi", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })).json();
  $("verify-btn").disabled = false; $("verify-btn").textContent = "Verify";
  if (d.ok) renderCredential(d);
  else $("cred-status").innerHTML = `<span class="pill no">${d.message}</span>`;
}

// --- wiring ---
["lasso", "brush", "eraser"].forEach((k) => $("tool-" + k).onclick = () => setTool(k));
$("brush-size").oninput = (e) => { state.brush = +e.target.value; $("size-val").textContent = e.target.value; };
$("confidence").oninput = (e) => { state.conf = +e.target.value; $("conf-val").textContent = e.target.value; };
$("brightness").oninput = $("contrast").oninput = () => {
  $("stage-wrap").style.filter = `brightness(${$("brightness").value}) contrast(${$("contrast").value})`;
};
document.querySelectorAll(".dx").forEach((b) => b.onclick = () => {
  state.dx = b.dataset.dx;
  document.querySelectorAll(".dx").forEach((x) => x.classList.toggle("active", x === b));
  updateLockEnabled();
});
$("undo").onclick = () => { const s = state.shapes.pop(); if (s) { s.destroy(); drawLayer.draw(); } updateLockEnabled(); };
$("clear").onclick = () => { state.shapes.forEach((s) => s.destroy()); state.shapes = []; drawLayer.draw(); updateLockEnabled(); };
$("lock").onclick = submit;
$("next").onclick = loadNextCase;
$("verify-btn").onclick = verifyNpi;

loadNextCase();
loadLeaderboard();
loadCredential();
