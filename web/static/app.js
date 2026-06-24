// MedVS-AI Phase 1 — vanilla JS + Konva. Draw the lesion -> lock -> reveal vs the model.
const DISPLAY = 480;        // stage size in px
const GRID = 256;           // server scoring grid; export at this resolution
const K = GRID / DISPLAY;   // image-space -> grid scale

// --- session ---
let sessionId = localStorage.getItem("medvs_session");
if (!sessionId) { sessionId = "s_" + Math.random().toString(36).slice(2) + Date.now().toString(36); localStorage.setItem("medvs_session", sessionId); }

// --- state ---
const state = { tool: "lasso", brush: 22, dx: null, conf: 50, locked: false, caseId: null,
  imageUrl: null, t0: 0, shapes: [], drawing: null, panMode: false,
  modality: "dermoscopy", diagnoses: [], expertBand: [0.75, 0.81], verifiedBadge: null };

// --- Konva ---
const stage = new Konva.Stage({ container: "stage", width: DISPLAY, height: DISPLAY });
const imgLayer = new Konva.Layer({ listening: false });
const drawLayer = new Konva.Layer();
stage.add(imgLayer); stage.add(drawLayer);

const $ = (id) => document.getElementById(id);
const pos = () => stage.getRelativePointerPosition();   // image-space, zoom/pan-invariant

function setTool(t) {
  state.tool = t;
  ["lasso", "brush", "eraser"].forEach((k) => $("tool-" + k).classList.toggle("active", k === t));
}
function setBrush(v) { state.brush = Math.max(4, Math.min(60, v)); $("brush-size").value = state.brush; $("size-val").textContent = state.brush; }

function updateLockEnabled() {
  const needDraw = state.shapes.length === 0, needDx = !state.dx;
  $("lock").disabled = state.locked || needDraw || needDx;
  let msg = "";
  if (state.locked) msg = "";
  else if (needDraw && needDx) msg = "Draw the lesion and pick a diagnosis to unlock.";
  else if (needDraw) msg = "Draw the lesion to unlock.";
  else if (needDx) msg = "Pick a diagnosis to unlock.";
  else msg = "Ready — lock to reveal. The model's answer stays hidden until you do.";
  $("lock-hint").textContent = msg;
}

// --- view: zoom + pan (view only; never affects stored/exported coords) ---
function clampView() {
  const s = stage.scaleX();
  if (s <= 1) { stage.scale({ x: 1, y: 1 }); stage.position({ x: 0, y: 0 }); return; }
  const min = -(s - 1) * DISPLAY;
  stage.position({ x: Math.min(0, Math.max(min, stage.x())), y: Math.min(0, Math.max(min, stage.y())) });
}
stage.on("wheel", (e) => {
  e.evt.preventDefault();
  const old = stage.scaleX(), ptr = stage.getPointerPosition();
  const m = { x: (ptr.x - stage.x()) / old, y: (ptr.y - stage.y()) / old };
  let s = e.evt.deltaY > 0 ? old / 1.1 : old * 1.1;
  s = Math.max(1, Math.min(8, s));
  stage.scale({ x: s, y: s });
  stage.position({ x: ptr.x - m.x * s, y: ptr.y - m.y * s });
  clampView(); stage.batchDraw();
});
stage.on("dragmove", clampView);
function setPan(on) { state.panMode = on; stage.draggable(on); $("stage-wrap").style.cursor = on ? "grab" : "crosshair"; }
function resetView() { stage.scale({ x: 1, y: 1 }); stage.position({ x: 0, y: 0 }); stage.batchDraw(); }

// --- case image ---
function loadCaseImage(url) {
  return new Promise((res) => {
    const im = new Image(); im.crossOrigin = "anonymous";
    im.onload = () => { imgLayer.destroyChildren(); imgLayer.add(new Konva.Image({ image: im, width: DISPLAY, height: DISPLAY })); imgLayer.draw(); res(); };
    im.src = url + "?t=" + Date.now();
  });
}
async function loadNextCase() {
  const r = await fetch(`/api/next_case?session_id=${encodeURIComponent(sessionId)}&modality=${encodeURIComponent(state.modality)}`);
  const data = await r.json();
  $("reveal-card").classList.remove("show");
  state.locked = false; state.dx = null; state.shapes = []; state.drawing = null;
  drawLayer.destroyChildren(); drawLayer.draw(); resetView();
  if (!data.case_id) { $("case-title").textContent = "No unseen cases for this modality — switch modality or seed more."; $("lock").disabled = true; $("lock-hint").textContent = ""; $("dx-buttons").innerHTML = ""; return; }
  state.caseId = data.case_id; state.imageUrl = data.image_url;
  state.expertBand = data.expert_dice_band || state.expertBand;
  renderDxButtons(data.diagnoses || []);
  $("case-title").textContent = `Case ${data.case_id} — trace ${data.draw_target || "the region"}`;
  await loadCaseImage(data.image_url);
  state.t0 = Date.now();
  updateLockEnabled();
}

// --- drawing ---
function strokeOpts() {
  if (state.tool === "eraser") return { stroke: "#000", strokeWidth: state.brush, lineCap: "round", lineJoin: "round", globalCompositeOperation: "destination-out" };
  if (state.tool === "brush") return { stroke: "rgba(229,57,53,0.55)", strokeWidth: state.brush, lineCap: "round", lineJoin: "round" };
  return { stroke: "rgba(229,57,53,0.9)", strokeWidth: 2, closed: false, fill: "rgba(229,57,53,0.4)" }; // lasso
}
stage.on("pointerdown", () => {
  if (state.locked || state.panMode) return;
  const p = pos();
  const line = new Konva.Line({ points: [p.x, p.y], ...strokeOpts() });
  drawLayer.add(line);
  state.drawing = { node: line, tool: state.tool, width: state.brush };
});
stage.on("pointermove", () => {
  if (!state.drawing) return;
  const p = pos();
  state.drawing.node.points(state.drawing.node.points().concat([p.x, p.y]));
  drawLayer.batchDraw();
});
stage.on("pointerup", () => {
  if (!state.drawing) return;
  if (state.drawing.tool === "lasso") state.drawing.node.closed(true);
  drawLayer.batchDraw();
  state.shapes.push(state.drawing); state.drawing = null;
  updateLockEnabled();
});
function undo() { const s = state.shapes.pop(); if (s) { s.node.destroy(); drawLayer.draw(); } updateLockEnabled(); }
function clearAll() { state.shapes.forEach((s) => s.node.destroy()); state.shapes = []; drawLayer.draw(); updateLockEnabled(); }

// Rasterize stored image-space strokes onto a 256x256 mask (transform-independent).
function exportMaskDataUrl() {
  const c = document.createElement("canvas"); c.width = GRID; c.height = GRID;
  const ctx = c.getContext("2d");
  ctx.lineCap = "round"; ctx.lineJoin = "round"; ctx.strokeStyle = "#fff"; ctx.fillStyle = "#fff";
  for (const s of state.shapes) {
    const pts = s.node.points(); if (pts.length < 4) continue;
    ctx.globalCompositeOperation = s.tool === "eraser" ? "destination-out" : "source-over";
    ctx.beginPath(); ctx.moveTo(pts[0] * K, pts[1] * K);
    for (let i = 2; i < pts.length; i += 2) ctx.lineTo(pts[i] * K, pts[i + 1] * K);
    if (s.tool === "lasso") { ctx.closePath(); ctx.fill(); }
    else { ctx.lineWidth = Math.max(1, s.width * K); ctx.stroke(); }
  }
  return c.toDataURL("image/png");
}

// --- submit / reveal ---
async function submit() {
  state.locked = true; $("lock").disabled = true; $("lock-hint").textContent = "";
  const body = { session_id: sessionId, case_id: state.caseId, badge: $("badge").value,
    diagnosis: state.dx, confidence: state.conf, mask_png: exportMaskDataUrl(), draw_ms: Date.now() - state.t0 };
  const r = await fetch("/api/attempt", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  renderReveal(await r.json(), body.mask_png);
  loadLeaderboard();
}
function fmt(v) { return v === null || v === undefined ? "—" : v; }
function renderReveal(d, youMask) {
  $("rv-image").src = state.imageUrl + "?t=" + Date.now();
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

// --- modality (registry-driven; diagnosis buttons are NOT hardcoded) ---
function renderDxButtons(diagnoses) {
  state.diagnoses = diagnoses; state.dx = null;
  const wrap = $("dx-buttons"); wrap.innerHTML = "";
  for (const d of diagnoses) {
    const b = document.createElement("button");
    b.className = "dx"; b.textContent = d.length <= 3 ? d.toUpperCase() : d.charAt(0).toUpperCase() + d.slice(1);
    b.onclick = () => { state.dx = d; wrap.querySelectorAll(".dx").forEach((x) => x.classList.toggle("active", x === b)); updateLockEnabled(); };
    wrap.appendChild(b);
  }
}
async function loadModalities() {
  const d = await (await fetch("/api/modalities")).json();
  const sel = $("modality"); sel.innerHTML = "";
  for (const m of d.modalities) {
    const o = document.createElement("option");
    o.value = m.id; o.textContent = m.label + (m.n_cases ? "" : " (no cases yet)");
    sel.appendChild(o);
  }
  state.modality = sel.value || "dermoscopy";
  sel.onchange = () => { state.modality = sel.value; loadNextCase(); };
}

// --- NPI credential verification ---
function renderCredential(d) {
  const el = $("cred-status");
  if (d.badge) {
    state.verifiedBadge = d.badge;
    const nm = d.name_match === false ? " (name didn't match — counted as unverified-name)" : "";
    el.innerHTML = `<span class="pill ok">✓ Verified: ${d.badge}${d.specialty ? " — " + d.specialty : ""}</span> Your attempts now count as <b>${d.badge}</b>.${nm}`;
    $("badge").disabled = true;
  } else { state.verifiedBadge = null; el.textContent = "Not verified — you'll play as your self-reported background."; $("badge").disabled = false; }
}
async function loadCredential() { renderCredential(await (await fetch(`/api/credential?session_id=${encodeURIComponent(sessionId)}`)).json()); }
async function verifyNpi() {
  const body = { session_id: sessionId, npi: $("npi").value.trim(), first_name: $("npi-first").value.trim(), last_name: $("npi-last").value.trim() };
  $("verify-btn").disabled = true; $("verify-btn").textContent = "Checking…";
  const d = await (await fetch("/api/verify_npi", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })).json();
  $("verify-btn").disabled = false; $("verify-btn").textContent = "Verify";
  if (d.ok) renderCredential(d); else $("cred-status").innerHTML = `<span class="pill no">${d.message}</span>`;
}

// --- wiring ---
["lasso", "brush", "eraser"].forEach((k) => $("tool-" + k).onclick = () => setTool(k));
$("brush-size").oninput = (e) => setBrush(+e.target.value);
$("confidence").oninput = (e) => { state.conf = +e.target.value; $("conf-val").textContent = e.target.value; };
$("brightness").oninput = $("contrast").oninput = () => {
  $("bright-val").textContent = (+$("brightness").value).toFixed(2);
  $("contrast-val").textContent = (+$("contrast").value).toFixed(2);
  $("stage-wrap").style.filter = `brightness(${$("brightness").value}) contrast(${$("contrast").value})`;
};
$("undo").onclick = undo;
$("clear").onclick = clearAll;
$("reset-view").onclick = resetView;
$("lock").onclick = submit;
$("next").onclick = loadNextCase;
$("verify-btn").onclick = verifyNpi;

// keyboard shortcuts (ignored while typing in inputs)
window.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  if (e.code === "Space" && !state.locked) { e.preventDefault(); if (!state.panMode) setPan(true); }
  else if (e.key === "[") setBrush(state.brush - 4);
  else if (e.key === "]") setBrush(state.brush + 4);
  else if (e.key === "z" || e.key === "Z") undo();
  else if (e.key === "1") setTool("lasso");
  else if (e.key === "2") setTool("brush");
  else if (e.key === "3") setTool("eraser");
});
window.addEventListener("keyup", (e) => { if (e.code === "Space") setPan(false); });
$("stage-wrap").style.cursor = "crosshair";

loadModalities().then(loadNextCase);
loadLeaderboard();
loadCredential();
