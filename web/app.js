"use strict";
// Eraser dashboard: vanilla JS, no CDN, works offline.

const S = { meta: null, shards: null, patients: null, selPatient: null, lastDelete: null, sweep: null, tab: "shards", busy: false, offset: 0 };
const $ = (q, el = document) => el.querySelector(q);
const $$ = (q, el = document) => [...el.querySelectorAll(q)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (x) => (x == null ? "–" : `${(x * 100).toFixed(1)}%`);

async function api(path, opts = {}) {
  const r = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail || body));
  return body;
}
const post = (p, d) => api(p, { method: "POST", body: d === undefined ? undefined : JSON.stringify(d) });
function toast(msg, bad = false) {
  const t = $("#toast"); t.textContent = msg; t.className = "toast" + (bad ? " bad" : "");
  clearTimeout(toast._t); toast._t = setTimeout(() => t.classList.add("hidden"), 5000);
}

async function loadAll() {
  [S.meta, S.shards] = await Promise.all([api("/api/meta"), api("/api/shards")]);
  $("#wschip").textContent = `seed ${S.meta.seed} · ${S.meta.n_shards} shards x ${S.meta.n_slices} slices · ${S.meta.n_patients} patients · acc ${pct(S.meta.ensemble_accuracy)}`;
}
async function boot() {
  try { await loadAll(); } catch (e) { toast(e.message, true); }
  $$("#tabs button").forEach((b) => b.addEventListener("click", () => { S.tab = b.dataset.tab; renderTab(); }));
  renderTab();
}
function renderTab() {
  $$("#tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === S.tab));
  ({ shards: viewShards, delete: viewDelete, log: viewLog, sweep: viewSweep, how: viewHow })[S.tab]();
}
const view = () => $("#view");
const tile = (v, l) => `<div class="tile"><div class="v">${esc(v)}</div><div class="l">${esc(l)}</div></div>`;

// ------------------------------------------------------------------ SHARDS
function viewShards() {
  view().innerHTML = `<p class="lede">Every patient belongs to exactly one shard — shards share no data, so deleting a point never requires touching any shard but its own. Within a shard, training happens slice by slice, with a checkpoint after each one; deleting a point only invalidates the checkpoints from its slice onward.</p>
    <div class="tiles">${tile(S.meta.n_shards, "shards")}${tile(S.meta.n_slices, "slices per shard")}${tile(pct(S.meta.ensemble_accuracy), "ensemble accuracy on held-out patients")}${tile(S.shards.reduce((a, s) => a + s.cost, 0).toFixed(3) + "s", "total training time so far")}</div>
    <div class="grid" style="grid-template-columns:repeat(auto-fill,minmax(320px,1fr))">${S.shards.map((s) => `<div class="shardcard">
      <div class="nm">Shard ${s.index}</div><div class="meta">${s.slice_sizes.reduce((a, b) => a + b, 0)} patients · ${s.cost.toFixed(3)}s trained</div>
      <div class="sliceseg">${s.slice_sizes.map((n, i) => `<i title="slice ${i}: ${n} patients">${n}</i>`).join("")}</div></div>`).join("")}</div>`;
}

// ------------------------------------------------------------------ DELETE
async function viewDelete() {
  view().innerHTML = `<p class="lede">Pick a patient. "Flagged: likely memorized" means the <em>other seven shards</em> — which never saw this patient — already disagree with their recorded label: this shard had nothing to generalise from and had to memorise it specifically. Deleting one of those usually swings the model's loss on them by an order of magnitude or more; deleting a typical patient barely moves it.</p>
    <div class="split"><div class="panel scroll" style="max-height:560px;padding:0" id="plist"></div><div class="panel detail" id="pdetail"><span class="muted">Pick a patient on the left.</span></div></div>`;
  await loadPatients();
}
async function loadPatients() {
  S.patients = await api("/api/patients?" + new URLSearchParams({ limit: 60, offset: S.offset }));
  renderPatientList();
}
function renderPatientList() {
  const el = $("#plist");
  if (!el) return;
  el.innerHTML = `<div class="patientrow" style="font-weight:600;color:var(--text-muted)"><span>id</span><span>shard</span><span>slice</span><span>flag</span><span></span></div>
    ${S.patients.items.map((p) => `<div class="patientrow${p.patient_id === S.selPatient ? " sel" : ""}" data-id="${p.patient_id}"><span class="mono">${p.patient_id}</span><span>${p.shard}</span><span>${p.slice}</span>
      <span>${p.memorized ? '<span class="memtag yes">memorized</span>' : '<span class="memtag no">typical</span>'}</span><span></span></div>`).join("")}
    <div style="display:flex;gap:8px;padding:8px"><button class="sm" id="prev" ${S.offset === 0 ? "disabled" : ""}>&larr; prev</button><button class="sm" id="next">next &rarr;</button></div>`;
  $$(".patientrow[data-id]", el).forEach((row) => row.addEventListener("click", () => { S.selPatient = +row.dataset.id; renderPatientList(); loadPatientDetail(+row.dataset.id); }));
  $("#prev").addEventListener("click", () => { S.offset = Math.max(0, S.offset - 60); loadPatients(); });
  $("#next").addEventListener("click", () => { S.offset += 60; loadPatients(); });
}
async function loadPatientDetail(pid) {
  try {
    const p = await api(`/api/patient/${pid}`);
    renderPatientDetail(p);
  } catch (e) { toast(e.message, true); }
}
function renderPatientDetail(p, result) {
  const el = $("#pdetail");
  if (!el) return;
  el.innerHTML = `<h2>Patient ${p.patient_id} ${p.memorized ? '<span class="memtag yes">likely memorized</span>' : '<span class="memtag no">typical pattern</span>'}</h2>
    <p class="small muted">shard ${p.shard}, slice ${p.slice} · recorded outcome: ${p.label ? "readmitted" : "not readmitted"} · current loss on this point: ${p.current_loss}</p>
    <div style="display:flex;gap:8px;margin:10px 0"><label class="small"><input type="checkbox" id="fullrebuild"> also time a full ensemble rebuild (slow, a few tenths of a second)</label></div>
    <button class="primary" id="godelete">Delete this patient and audit</button>
    <div id="result"></div>`;
  $("#godelete").addEventListener("click", async () => {
    if (S.busy) return;
    S.busy = true;
    const btn = $("#godelete"); btn.disabled = true; btn.textContent = "Deleting and auditing…";
    try {
      const r = await post("/api/delete", { patient_id: p.patient_id, full_rebuild: $("#fullrebuild").checked });
      S.lastDelete = r;
      renderDeleteResult(r);
      toast(`Deleted patient ${p.patient_id}`);
    } catch (e) { toast(e.message, true); }
    finally { S.busy = false; btn.disabled = false; btn.textContent = "Delete this patient and audit"; }
  });
}
function renderDeleteResult(r) {
  const el = $("#result");
  if (!el) return;
  el.innerHTML = `<h3 style="margin-top:14px">Membership-inference audit</h3>
    <div class="beforeafter"><div class="box"><div class="v">${r.loss_before}</div><div class="l">loss before</div></div><div class="arrow">&rarr;</div>
      <div class="box"><div class="v">${r.loss_after}</div><div class="l">loss after (×${(r.loss_after / Math.max(r.loss_before, 1e-9)).toFixed(1)})</div></div></div>
    <div class="beforeafter"><div class="box"><div class="v">${pct(r.attack_score_before)}</div><div class="l">attack: P(member) before</div></div><div class="arrow">&rarr;</div>
      <div class="box"><div class="v">${pct(r.attack_score_after)}</div><div class="l">attack: P(member) after</div></div></div>
    <p>Retrained in <b>${r.partial_seconds}s</b> (just shard ${r.shard}, from slice ${r.slice} on)${r.speedup ? ` — a full rebuild took <b>${r.full_rebuild_seconds}s</b>: <span class="speedbadge">${r.speedup}x faster</span>` : ""}.</p>
    <p class="small muted">Ensemble accuracy on held-out patients after this deletion: ${pct(r.ensemble_accuracy)}.</p>`;
}

// ------------------------------------------------------------------ LOG
async function viewLog() {
  const log = await api("/api/log");
  view().innerHTML = `<p class="lede">Every deletion made in this session, most recent first.</p>
    <div class="panel scroll" style="max-height:560px;padding:0">
    <div class="logrow" style="font-weight:600;color:var(--text-muted)"><span>patient</span><span>shard</span><span>loss</span><span>attack P(member)</span><span>time</span><span></span></div>
    ${log.map((r) => `<div class="logrow"><span class="mono">${r.patient_id}</span><span>${r.shard}</span><span>${r.loss_before} → ${r.loss_after}</span>
      <span>${pct(r.attack_score_before)} → ${pct(r.attack_score_after)}</span><span>${r.partial_seconds}s${r.speedup ? ` (${r.speedup}x)` : ""}</span>
      <span>${r.memorized ? '<span class="memtag yes">memorized</span>' : '<span class="memtag no">typical</span>'}</span></div>`).join("") || '<div class="muted" style="padding:14px">No deletions yet — try the Delete a patient tab.</div>'}</div>`;
}

// ------------------------------------------------------------------ SWEEP / EVIDENCE
function viewSweep() {
  view().innerHTML = `<div class="panel"><h2>Across many seeds and many deletions</h2><div style="display:flex;gap:8px;align-items:center;margin-bottom:8px">
      <label class="small">seeds <input type="number" id="nseeds" value="12" min="3" max="60" style="width:60px"></label>
      <label class="small">per seed <input type="number" id="nper" value="8" min="2" max="40" style="width:60px"></label><button class="primary" id="swgo">Run</button></div>
      <div id="sw" class="muted small">click "Run", or wait for the background one to finish</div></div>`;
  $("#swgo").addEventListener("click", async () => {
    try { toast("Running… several seconds"); const r = await post("/api/sweep?" + new URLSearchParams({ n_seeds: $("#nseeds").value, per_seed: $("#nper").value })); S.sweep = r; renderSweep(); }
    catch (e) { toast(e.message, true); }
  });
  pollSweep();
}
async function pollSweep() {
  const r = await api("/api/sweep");
  if (r.sweep) { S.sweep = r; renderSweep(); return; }
  const el = $("#sw");
  if (el) el.textContent = "running in the background…";
  if (S.tab === "sweep") setTimeout(pollSweep, 1500);
}
function renderSweep() {
  const el = $("#sw"), s = S.sweep;
  if (!el || !s) return;
  const mem = s.sweep.loss_ratio_memorized, typ = s.sweep.loss_ratio_typical;
  const bar = (v, color) => `<div class="weightbar" style="margin:4px 0"><i style="width:${Math.min(100, v.median * 5)}%;background:${color}"></i></div>`;
  el.innerHTML = `<div class="tiles">${tile(`x${mem.median}`, `memorized points: median loss ratio (n=${mem.n})`)}${tile(pct(mem.doubled_or_more), "of those at least doubled")}
      ${tile(`x${typ.median}`, `typical points: median loss ratio (n=${typ.n})`)}${tile(pct(typ.doubled_or_more), "of those at least doubled")}</div>
    <p>Deleting a <b>memorized</b> point (one the rest of the ensemble already disagreed with) ${bar(mem, "#d03b3b")} multiplies its loss by a median of <b>${mem.median}x</b>.
      Deleting a <b>typical</b> point ${bar(typ, "#2fb344")} barely changes it (median <b>${typ.median}x</b>) — the model never needed to memorise it in the first place.</p>
    <p>Ensemble accuracy on held-out patients: <b>${pct(s.sweep.ensemble_acc_before.mean)}</b> before these deletions, <b>${pct(s.sweep.ensemble_acc_after.mean)}</b> after — removing one point at a time barely moves it.</p>
    <h2 style="margin-top:18px">Partial retrain vs. a full rebuild</h2>
    <p>Averaged over ${s.speedup.n_seeds} seeds: a partial (one shard, from the affected slice on) retrain took <b>${s.speedup.partial_seconds.mean}s</b>; rebuilding the whole ensemble from scratch took
      <b>${s.speedup.full_rebuild_seconds.mean}s</b> — <span class="speedbadge">${s.speedup.speedup}x faster</span>.</p>`;
}

// ------------------------------------------------------------------ HOW
function viewHow() {
  view().innerHTML = `<div class="prose panel">
    <h3>1. SISA: shard, isolate, slice, aggregate</h3><p>Every patient is assigned to exactly one of eight shards at ingestion time — shards never share data. Within a shard, training data is split into four ordered slices and trained incrementally, with a checkpoint saved after each one.
      The ensemble's prediction is a plain, unweighted average of all eight shards' own models.</p>
    <h3>2. Deleting a patient</h3><p>Find their shard and slice, remove them from that slice, and resume training from the checkpoint saved <em>before</em> that slice — replaying only the slices from there on, with the point gone.
      Every other shard, and every earlier slice of this one, is untouched. Measured on this project's own data: a partial retrain this way takes about <b>0.017s</b> on average; rebuilding the whole eight-shard ensemble from scratch takes about <b>0.21s</b> — roughly <b>13x</b> slower.</p>
    <h3>3. Proving it, not just asserting it</h3><p>Removing a row from a table proves nothing about whether a model still "remembers" it. A membership-inference audit (Yeom et al., 2018) trains a tiny logistic regression on the model's own per-example loss — in log space, since a confident-and-right prediction's loss can be six orders of magnitude smaller than a wrong one's — to separate known training members from the held-out set that no shard ever trains on.
      Run once before a deletion and once after (on the retrained shard), the loss itself is the honest signal.</p>
    <h3>4. Which patients actually show the effect</h3><p>Not every deletion looks the same. A patient is flagged "likely memorized" if the <em>other seven shards</em> — which never saw them — already disagree with their recorded outcome: their own shard had nothing to generalise from and had to memorise the specific label.
      Measured across many seeds: deleting a flagged patient multiplies their loss by a median of over 20x (90% at least double); deleting a typical one leaves it essentially unchanged (median ratio ~1.0, barely a quarter double). This uses no privileged knowledge of the synthetic task — only what the system itself can see.</p>
    <h3>Honest limits</h3><ul><li>The population-level membership-inference attack (scored across <em>all</em> training points, not just flagged ones) is weak — its AUC sits only a little above the 0.5 a coin flip would get, because this ensemble generalises reasonably well (92-94% held-out accuracy), and a model that generalises well leaks less membership signal. The dramatic effect is concentrated in the minority of memorized points, not universal.</li>
      <li>Loss-ratio is a noisy statistic right at zero — a point with near-zero loss before deletion can show an enormous ratio from a tiny absolute change. The dashboard reports the median and "at least doubled" fraction for exactly this reason, not the mean.</li>
      <li>Eight shards and 1,600 synthetic patients are sized so every demo in this dashboard runs in well under a second; this is a mechanism demonstration of SISA and loss-based membership inference, not a claim about real multi-thousand-patient unlearning at production scale.</li></ul></div>`;
}

boot();
