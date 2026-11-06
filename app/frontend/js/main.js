import { get } from './api.js';
import { esc, badge, badges, validityBadge, unavailable, experimentDetail, fmt } from './ui.js';
import { renderSim } from './sim.js';
import { renderGraph, nodePanel } from './arch.js';
const VIEWS = [['overview', 'Overview', '▣'], ['simulation', 'Simulation', '▶'], ['experiments', 'Experiments', '☰'], ['act', 'ACT Policy', '⚙'], ['worldmodel', 'World Model', '◎'], ['architecture', 'Architecture', '⬡'], ['future', 'Future WAM', '✦'], ['status', 'Research Status', '⚑']];
const app = document.getElementById('app'), main = document.getElementById('main');
let health = null, modes = [], status = null;
function setTop(o) { const s = (id, v) => { const e = document.getElementById(id); if (e && v !== undefined) e.innerHTML = v; }; if (o.exp !== undefined) s('t-exp', esc(o.exp)); if (o.ckpt !== undefined) s('t-ckpt', esc(o.ckpt)); if (o.ev !== undefined) s('t-ev', Array.isArray(o.ev) ? badges(o.ev) : (o.ev ? `<span class="small">${esc(o.ev)}</span>` : '—')); if (o.src !== undefined) s('t-src', esc(o.src)); }
function topSim() { const r = health && health.render; const e = document.getElementById('t-sim'); if (!r) { e.textContent = 'API unreachable'; return; } e.innerHTML = r.error ? `<span class="no">render unavailable</span>` : (r.initialised ? '<span class="ok">MuJoCo scene loaded (read-only)</span>' : 'MuJoCo idle (loads on first frame)'); e.title = r.error || ''; }
async function boot() {
  document.getElementById('navlinks').innerHTML = VIEWS.map(([id, n, i]) => `<a data-v="${id}"><i>${i}</i><span>${n}</span></a>`).join('');
  const [h, m, s] = await Promise.all([get('/api/health'), get('/api/modes'), get('/api/research/status')]); health = h.ok ? h.data : null; modes = m.ok ? m.data : []; status = s.ok ? s.data : null; topSim();
  document.getElementById('navhead').textContent = health ? `HEAD ${health.repo_head || '?'} · v${health.version}` : 'API unreachable';
  window.addEventListener('hashchange', route); route();
}
async function route() {
  const id = (location.hash.replace('#/', '') || 'overview').split('/')[0]; const v = VIEWS.find(x => x[0] === id) ? id : 'overview'; app.dataset.view = v;
  document.querySelectorAll('#navlinks a').forEach(a => { a.classList.toggle('on', a.dataset.v === v); a.onclick = () => { location.hash = '#/' + a.dataset.v; }; });
  main.scrollTop = 0; main.innerHTML = '<p class="muted">Loading…</p>';
  if (!health) { main.innerHTML = unavailable('The API did not answer /api/health. Start the server with: python -m app.backend.server', 'Backend'); return; }
  ({ overview, simulation, experiments, act, worldmodel, architecture, future, status: statusView })[v]();
}
async function overview() {
  setTop({ exp: '—', ev: '—', src: 'saved artifacts under outputs/ (read-only)', ckpt: 'ACT c0_s0 = e6b_c0_seed0' });
  const ex = status ? status.experiments : [];
  main.innerHTML = `${health.data && health.data.message && !health.static_snapshot ? `<div class="snapbar"><b>DATA NOTE</b> ${String(health.data.message).replace(/[<>&]/g, c => ({ '<': '&lt;', '>': '&gt;', '&': '&amp;' }[c]))}</div>` : ''}${health.static_snapshot ? '<div class="snapbar"><b>STATIC SNAPSHOT</b> of the running app: the API answers are embedded, three episodes are pre-rendered (one Stage-1 trajectory and one WM-1 episode reconstructed in MuJoCo, one dataset episode with its stored camera). Free orbit and live rendering need the real server.</div>' : ''}<div class="hero"><h1>OpenArm Research Studio</h1><p class="lead">Inspect the completed OpenArm v1 + MuJoCo + ACT + World Model research: replay saved robot trajectories, read every experiment with its provenance, explore the current architecture and preview the proposed future World Action Model. This is a read-only viewer: it does not run experiments, policies or training.</p>
  <div class="legend small muted"><span>${badge('IMPLEMENTED')} code exists</span><span>${badge('VALIDATED')} independently validated, all gates PASS</span><span>${badge('EXPERIMENTAL')} development evidence</span><span>${badge('DESIGN ONLY')} not built</span><span>${badge('NOT TRAINED')}</span><span>${badge('NOT EVALUATED')}</span></div></div>
  <h2>Research stages</h2><div class="grid g3">${ex.map(e => `<div class="card click" data-go="${e.id === 'wam' ? 'future' : 'experiments/' + e.id}"><div>${badges(e.labels)}</div><h3>${esc(e.title)}</h3><p class="small muted">${esc(e.purpose)}</p>${e.validity ? `<p>${validityBadge(e.validity)}</p>` : ''}</div>`).join('')}</div>
  <h2>Workspaces</h2><div class="grid g3"><div class="card click" data-go="simulation"><h3>Simulation workspace</h3><p class="small muted">Recorded replay (A) and MuJoCo scene view (B). Live preview (C) is disabled and requires separate authorisation.</p></div><div class="card click" data-go="architecture"><h3>Architecture explorer</h3><p class="small muted">Current implementation, Stage-1 historical flow and the G2b grasp-correction flow are shown as separate layers.</p></div><div class="card click" data-go="future"><h3>Future WAM lab</h3><p class="small muted">Design-only preview. No trained model, no predictions.</p></div></div>
  <h2>Reading guide</h2><div class="banner info">Numbers shown in the Experiments views are read from the saved artifacts when you open a card, with the artifact path and a per-metric provenance tag. Anything illustrative is stamped <b>SYNTHETIC ILLUSTRATION &mdash; NOT MODEL OUTPUT</b>.</div>`;
  main.querySelectorAll('[data-go]').forEach(c => c.onclick = () => location.hash = '#/' + c.dataset.go);
}
async function simulation() { setTop({ ckpt: 'ACT c0_s0 = e6b_c0_seed0 (not loaded by this view)' }); await renderSim(main, health, setTop, modes); }
async function experiments() {
  const parts = location.hash.replace('#/', '').split('/'); const r = await get('/api/experiments'); if (!r.ok) { main.innerHTML = unavailable(r.reason, 'Experiments'); return; }
  const cur = parts[1];
  main.innerHTML = `<h1>Experiment explorer</h1><p class="lead">Each card reads its metrics from saved artifacts. Statuses are derived from the artifacts, not asserted.</p><div class="grid g3">${r.data.map(e => `<div class="card click ${cur === e.id ? 'sel' : ''}" data-id="${e.id}" style="${cur === e.id ? 'border-color:var(--cyan)' : ''}"><div>${badges(e.labels)}</div><h3>${esc(e.title)}</h3><p class="small muted">${esc(e.purpose)}</p>${validityBadge(e.validity)}${e.unavailable ? `<span class="badge b-UNAV">${e.unavailable} metric(s) unavailable</span>` : ''}</div>`).join('')}</div><div id="detail" style="margin-top:18px"></div>`;
  main.querySelectorAll('.card.click').forEach(c => c.onclick = () => location.hash = '#/experiments/' + c.dataset.id);
  if (cur) { const d = await get('/api/experiments/' + encodeURIComponent(cur)); const box = document.getElementById('detail'); if (!d.ok) { box.innerHTML = unavailable(d.reason, 'Experiment'); return; } setTop({ exp: d.data.title, ev: d.data.labels, src: 'saved artifacts (see provenance)' }); box.innerHTML = experimentDetail(d.data) + extra(d.data); box.scrollIntoView({ behavior: 'smooth' }); }
}
function extra(c) {
  if (c.id !== 'g2b') return '';
  return `<h2>Candidates (design)</h2><div class="grid g4">${c.candidate_defs.map(x => `<div class="card"><h3>${esc(x.id)} · ${esc(x.name)}</h3><p class="small muted">${esc(x.kind)}</p><p class="small">${esc(x.text)}</p>${x.oracle ? '<span class="badge b-INVALID">ORACLE INPUT</span>' : ''}</div>`).join('')}</div>
  <h2>Event-based endpoint logic</h2><table><tr><th>Event</th><th>Kind</th><th>Definition</th></tr>${c.endpoint_events.map(e => `<tr><td class="mono">${esc(e.id)}</td><td>${esc(e.kind)}</td><td>${esc(e.text)}</td></tr>`).join('')}</table><div class="grid g3" style="margin-top:10px">${c.endpoints.map(e => `<div class="card"><b>${esc(e.id)}</b> ${esc(e.name)}</div>`).join('')}</div>
  <h2>Integration self-tests (not an evaluation)</h2><table><tr><th>Gate</th><th>Result</th><th>CPU [s]</th><th>Source</th></tr>${c.integration_gates.map(g => `<tr><td>${esc(g.gate)}</td><td class="${g.result === 'PASS' ? 'ok' : 'no'}">${esc(g.result)}</td><td class="num">${fmt(g.cpu_s)}</td><td class="mono small">${esc(g.source || g.reason || '')}</td></tr>`).join('')}</table>`;
}
async function act() {
  const r = await get('/api/experiments/act_v1'); if (!r.ok) { main.innerHTML = unavailable(r.reason, 'ACT card'); return; } const c = r.data; setTop({ exp: c.title, ev: c.labels, ckpt: 'ACT V1 (act_full_v1) / c0_s0 (e6b_c0_seed0)', src: 'outputs/audit_v1, outputs/e6b' });
  const rows = (c.arms_table || []).map(a => `<tr><td>${esc(a.arm)}</td><td class="num">${a.nom ? a.nom[0] + '/' + a.nom[1] : '—'}</td><td class="num">${a.bound ? a.bound[0] + '/' + a.bound[1] : '—'}</td><td class="num">${fmt(a.dz_mean_cm, 2)}</td><td class="num">${fmt(a.jerk_all, 1)}</td></tr>`).join('');
  main.innerHTML = `<h1>ACT policy</h1><div class="banner warn">Historical, development evidence in simulation. Checkpoints are only described here; no model is loaded by this application.</div>${experimentDetail(c)}<h2>Stage 0/1 re-training comparison</h2><table><tr><th>Arm</th><th>NOM success</th><th>BOUND success</th><th>closing height [cm]</th><th>jerk [rad/s³]</th></tr>${rows}</table><p class="small muted mono">source: outputs/e6b/stage1_results.json (arms V1, C0, S1, S2)</p>`;
}
async function worldmodel() {
  const r = await get('/api/experiments/wm3b'); if (!r.ok) { main.innerHTML = unavailable(r.reason, 'WM3-B card'); return; } const c = r.data; setTop({ exp: c.title, ev: c.labels, ckpt: 'WM3-B (outputs/wm3/models/wm3b.pt)', src: 'outputs/indep/evaluation_independent.json' });
  const curve = c.curve_box_mm || {}, ks = ['K=1', 'K=5', 'K=10', 'K=20']; const all = Object.values(curve).flatMap(o => Object.values(o)); const mx = Math.max(1, ...all); const cols = { wm3b: '#41d8ea', wm2_c0: '#f0b45a', persistence: '#7b8fb0', constvel: '#9d8cf5' };
  const pts = o => ks.map((k, i) => `${60 + i * 150},${190 - 160 * (o[k] || 0) / mx}`).join(' ');
  const svg = `<svg viewBox="0 0 700 220" width="100%" style="max-width:760px"><rect x="40" y="10" width="640" height="190" fill="none" stroke="#1a2a47"/>${ks.map((k, i) => `<text x="${60 + i * 150}" y="214" fill="#7b8fb0" font-size="11" text-anchor="middle">${k}</text>`).join('')}${Object.entries(curve).map(([m, o]) => `<polyline points="${pts(o)}" fill="none" stroke="${cols[m] || '#fff'}" stroke-width="2"/><text x="${60 + 3 * 150 + 6}" y="${190 - 160 * (o['K=20'] || 0) / mx}" fill="${cols[m] || '#fff'}" font-size="11">${esc(m)} ${fmt(o['K=20'], 2)}</text>`).join('')}<text x="44" y="22" fill="#7b8fb0" font-size="10">box position error [mm] (50 independent groups)</text></svg>`;
  const cal = c.calibration ? `<table><tr><th>Target</th><th>AUROC</th><th>ECE</th><th>Brier</th><th>Evaluable</th></tr>${Object.entries(c.calibration).map(([k, v]) => `<tr><td class="mono">${esc(k)}</td><td class="num">${fmt(v.auroc, 3)}</td><td class="num">${fmt(v.ece, 3)}</td><td class="num">${fmt(v.brier, 3)}</td><td>${v.evaluable ? 'yes' : 'no'}</td></tr>`).join('')}</table>` : unavailable('calibration table missing', 'Calibration');
  main.innerHTML = `<h1>World model WM3-B</h1><div class="banner warn">Offline prediction errors on privileged simulator state. Not task success; no real-robot sensing path.</div>${experimentDetail(c)}<h2>Box position error vs horizon</h2>${svg}<h2>Classification calibration caveat</h2>${cal}`;
}
async function architecture() {
  const r = await get('/api/architecture/current'); if (!r.ok) { main.innerHTML = unavailable(r.reason, 'Architecture'); return; } const d = r.data; let layer = 'current', sel = null; setTop({ exp: 'Architecture explorer', ev: ['IMPLEMENTED'], src: 'repository code + saved artifacts' });
  const draw = () => {
    const L = d.layers.find(l => l.id === layer), nodes = d.nodes.filter(n => n.layers.includes(layer)), ids = new Set(nodes.map(n => n.id)), edges = d.edges.filter(e => e.layers.includes(layer) && ids.has(e.s) && ids.has(e.t));
    main.innerHTML = `<h1>Architecture explorer</h1><div class="tabs">${d.layers.map(l => `<button data-l="${l.id}" class="${l.id === layer ? 'on' : ''}">${esc(l.name.split(' - ')[0])}</button>`).join('')}<button id="tofuture">PROPOSED FUTURE ARCHITECTURE →</button></div><div class="banner ${layer === 'current' ? 'info' : 'warn'}"><b>${esc(L.name)}</b><br>${esc(L.text)}</div><div class="grid" style="grid-template-columns:1fr 340px"><div class="card" id="gr"></div><div id="np"></div></div><p class="small muted">All implementation files were checked for existence by the backend. Dashed borders = not implemented.</p>`;
    renderGraph(document.getElementById('gr'), nodes, edges, id => { sel = id; draw(); }, sel); document.getElementById('np').innerHTML = nodePanel(d.nodes.find(n => n.id === sel) && Object.assign({}, d.nodes.find(n => n.id === sel), { status: d.nodes.find(n => n.id === sel).status }));
    main.querySelectorAll('[data-l]').forEach(b => b.onclick = () => { layer = b.dataset.l; sel = null; draw(); }); document.getElementById('tofuture').onclick = () => location.hash = '#/future';
  };
  draw();
}
async function future() {
  const r = await get('/api/architecture/future'); if (!r.ok) { main.innerHTML = unavailable(r.reason, 'Future architecture'); return; } const d = r.data; let tab = 'A', sel = 'gwm'; setTop({ exp: 'Future WAM (proposed)', ev: ['DESIGN ONLY', 'NOT TRAINED'], src: 'design only - no artifacts', ckpt: 'none (no WAM checkpoint exists)' });
  const draw = () => {
    const tabs = [['A', 'Architecture flow'], ['B', 'Model inspector'], ['C', 'Candidate comparison'], ['D', 'Prediction timeline'], ['E', 'Sim2Real readiness']];
    main.innerHTML = `<h1>Future World Action Model lab</h1><div class="banner"><b>${esc(d.banner)}</b></div><div class="tabs">${tabs.map(([k, n]) => `<button data-t="${k}" class="${tab === k ? 'on' : ''}">${k} · ${n}</button>`).join('')}</div><div id="tabbody"></div>`;
    const tb = document.getElementById('tabbody');
    if (tab === 'A') { tb.innerHTML = `<div class="grid" style="grid-template-columns:1fr 340px"><div class="card" id="gr"></div><div id="np"></div></div>`; renderGraph(document.getElementById('gr'), d.nodes.map(n => ({ ...n, layers: ['future'] })), d.edges, id => { sel = id; draw(); }, sel); document.getElementById('np').innerHTML = nodePanel(d.nodes.find(n => n.id === sel)); }
    if (tab === 'B') { const n = d.nodes.find(x => x.id === sel); tb.innerHTML = `<div class="toolbar"><select id="nsel">${d.nodes.map(x => `<option value="${x.id}" ${x.id === sel ? 'selected' : ''}>${esc(x.label)}</option>`).join('')}</select></div>${nodePanel(n)}<div class="banner warn">Tensor shapes on this page are PROPOSED representations, not implemented or verified.</div>`; document.getElementById('nsel').onchange = e => { sel = e.target.value; draw(); }; }
    if (tab === 'C') tb.innerHTML = `<div class="banner warn">Illustrative cards only: no predicted probability or success improvement exists for any candidate.</div><div class="grid g4">${d.candidate_cards.map(c => `<div class="card"><h3>${esc(c.id)}</h3><p class="small">${esc(c.text)}</p><span class="badge b-NOTEVAL">NO PREDICTION AVAILABLE</span>${/oracle|true box/i.test(c.text) ? '<span class="badge b-INVALID">ORACLE INPUT</span>' : ''}</div>`).join('')}</div>`;
    if (tab === 'D') { tb.innerHTML = `<div class="card" style="position:relative"><div class="wm">${esc(d.timeline_illustration.label)}</div><canvas id="ptl" width="900" height="220" style="width:100%"></canvas><p class="small muted">${esc(d.timeline_illustration.label)}: the bars only show WHEN each event type could be predicted along a conceptual episode. They carry no probabilities.</p></div>`; const c = document.getElementById('ptl').getContext('2d'); const ev = d.timeline_illustration.events, pos = [[0.22, 0.30], [0.30, 0.62], [0.36, 0.40], [0.52, 0.70], [0.74, 0.86], [0.86, 0.98]]; c.font = '12px sans-serif'; ev.forEach((e, i) => { const y = 24 + i * 32; c.fillStyle = '#7b8fb0'; c.fillText(e, 6, y + 12); c.fillStyle = 'rgba(157,140,245,.35)'; c.fillRect(170 + pos[i][0] * 700, y, (pos[i][1] - pos[i][0]) * 700, 18); c.strokeStyle = '#9d8cf5'; c.setLineDash([4, 3]); c.strokeRect(170 + pos[i][0] * 700, y, (pos[i][1] - pos[i][0]) * 700, 18); c.setLineDash([]); }); }
    if (tab === 'E') tb.innerHTML = `<table><tr><th>Simulator-only dependency</th><th>Status on the real robot</th><th>Where it is used</th></tr>${d.sim2real.map(s => `<tr><td>${esc(s.item)}</td><td><span class="badge b-INVALID">${esc(s.status)}</span></td><td class="small">${esc(s.note)}</td></tr>`).join('')}</table>`;
    main.querySelectorAll('[data-t]').forEach(b => b.onclick = () => { tab = b.dataset.t; draw(); });
  };
  draw();
}
async function statusView() {
  setTop({ exp: 'Research status', ev: '—', src: 'derived from artifacts and design documents' }); if (!status) { main.innerHTML = unavailable('status endpoint failed', 'Research status'); return; }
  main.innerHTML = `<h1>Research status</h1><p class="lead">What exists, what was validated, and what is still open. Labels are computed from artifact content and file existence.</p><table><tr><th>Stage</th><th>Labels</th><th>Outcome validity</th></tr>${status.experiments.map(e => `<tr><td>${esc(e.title)}</td><td>${badges(e.labels)}</td><td>${validityBadge(e.validity)}</td></tr>`).join('')}</table>
  <h2>Open blockers (verified in the design documents)</h2>${status.blockers.map(b => `<div class="card" style="margin-bottom:8px"><span class="${b.verified_in_doc ? 'ok' : 'no'}">${b.verified_in_doc ? '✓ confirmed in' : (b.doc_missing ? '○ document not in this checkout (unverified):' : '✗ not confirmed in')}</span> <span class="mono small">${esc(b.source)}</span><br>${esc(b.text)}</div>`).join('')}
  <h2>Application modes</h2><table>${status.modes.map(m => `<tr><td>${esc(m.id)}</td><td>${esc(m.name)}</td><td class="${m.enabled ? 'ok' : 'no'}">${m.enabled ? 'enabled' : 'disabled'}</td><td class="small">${esc(m.text)}</td></tr>`).join('')}</table><h2>Rules</h2><ul>${status.rules.map(x => `<li>${esc(x)}</li>`).join('')}</ul>`;
}
boot();
