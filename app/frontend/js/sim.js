// Simulation workspace: MODE A (recorded replay), MODE B (MuJoCo scene view), MODE C (disabled).  Frames come from saved camera images or from a read-only MuJoCo reconstruction (mj_forward only).
import { get, frameUrl, sceneUrl } from './api.js';
import { esc, fmt, unavailable } from './ui.js';
const DT = 0.05;
export const S = { episodes: [], sel: null, tel: null, k: 0, playing: false, speed: 1, view: 'mujoco:overview', mode: 'A', orbit: { az: 140, el: -25, dist: 1.9 }, cache: new Map(), loading: new Set(), health: null, fk: null, err: null, timer: null, token: 0 };
let host, setTop;
export function frameIndexAt(tSeconds, n) { return Math.max(0, Math.min(n - 1, Math.floor(tSeconds / DT + 1e-9))); }   // pure helper, unit-tested
const urlFor = k => S.mode === 'B' ? sceneUrl(S.view.replace('mujoco:', ''), S.orbit) : frameUrl(S.sel.id, k, S.view, S.orbit);
const keyFor = k => `${S.mode}|${S.sel ? S.sel.id : 'scene'}|${k}|${S.view}|${S.view.endsWith('free') || S.mode === 'B' ? Math.round(S.orbit.az) + ',' + Math.round(S.orbit.el) + ',' + S.orbit.dist.toFixed(1) : ''}`;
async function loadFrame(k) {
  const key = keyFor(k); if (S.cache.has(key)) return S.cache.get(key); if (S.loading.has(key)) return null; S.loading.add(key);
  try {
    const r = await fetch(urlFor(k)); if (!r.ok) { let j = {}; try { j = await r.json(); } catch { j = {}; } S.err = j.reason || r.statusText; return null; }
    S.err = null; const u = URL.createObjectURL(await r.blob()); S.cache.set(key, u); if (S.cache.size > 260) { const f = S.cache.keys().next().value; URL.revokeObjectURL(S.cache.get(f)); S.cache.delete(f); } return u;
  } catch (e) { S.err = String(e); return null; } finally { S.loading.delete(key); }
}
async function prefetch(k) { const n = S.tel ? S.tel.n_frames : 1; for (let i = 1; i <= 6 && k + i < n; i++) { if (S.loading.size >= 3) break; loadFrame(k + i); } }
async function show(k) {
  const tok = ++S.token; S.k = k; const img = host.querySelector('#frame'); const hud = host.querySelector('#hud'), hud2 = host.querySelector('#hud2');
  if (hud) hud.textContent = S.tel ? `frame ${k + 1}/${S.tel.n_frames}   t = ${(k * DT).toFixed(2)} s` : 'scene (home pose)';
  const u = await loadFrame(k); if (tok !== S.token) return;
  if (u) { img.src = u; img.style.display = ''; host.querySelector('#unavail').innerHTML = ''; hud2.textContent = ''; prefetch(k); }
  else { hud2.textContent = S.loading.size ? 'rendering…' : ''; if (S.err) { host.querySelector('#unavail').innerHTML = unavailable(S.err, 'Frame'); } }
  drawInspector(); drawTimeline(); host.querySelector('#seek').value = k; host.querySelector('#tcur').textContent = `${(k * DT).toFixed(2)} s`;
  clearTimeout(S.fkT); if (S.tel && S.health && !S.health.render.error) S.fkT = setTimeout(async () => { const r = await get(`/api/episodes/${encodeURIComponent(S.sel.id)}/fk/${k}`); S.fk = r.ok ? r.data : { error: r.reason }; drawInspector(); }, 350);
}
function play() {
  if (!S.tel) return; S.playing = true; host.querySelector('#pp').textContent = '❚❚ Pause';
  const step = async () => { if (!S.playing) return; const nxt = S.k + 1; if (nxt >= S.tel.n_frames) { stop(); return; } const key = keyFor(nxt); if (!S.cache.has(key)) { host.querySelector('#hud2').textContent = 'buffering…'; loadFrame(nxt); S.timer = setTimeout(step, 120); return; } await show(nxt); S.timer = setTimeout(step, DT * 1000 / S.speed); };
  step();
}
function stop() { S.playing = false; clearTimeout(S.timer); const b = host.querySelector('#pp'); if (b) b.textContent = '▶ Play'; }
export async function renderSim(h, health, setTopFn, modes) {
  host = h; setTop = setTopFn; S.health = health; stop();
  const r = await get('/api/episodes'); S.episodes = r.ok ? r.data : []; const groups = [...new Set(S.episodes.map(e => e.source + ':' + e.group))];
  const renderOk = health && health.render && !health.render.error;
  host.innerHTML = `<h1>Simulation workspace</h1><p class="lead">Replay saved trajectories or inspect the OpenArm v1 MuJoCo scene. Frames are reconstructed with <span class="mono">mj_forward</span> only: nothing is simulated, trained or evaluated here.</p>
  <div class="toolbar"><span class="muted small">MODE</span>${modes.map(m => `<button data-mode="${m.id}" ${m.enabled ? '' : 'disabled'} title="${esc(m.text)}" class="${S.mode === m.id ? 'on' : ''}">${m.id} · ${esc(m.name)}</button>`).join('')}</div>
  <div class="toolbar" id="srcbar"><select id="grp">${groups.map(g => `<option value="${esc(g)}">${esc(g)}</option>`).join('')}</select><select id="ep" style="min-width:300px"></select><select id="viewsel"></select><select id="spd"><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select></div>
  <div class="viewer" id="vwr"><img id="frame" alt="rendered frame" style="display:none"><div class="hud" id="hud"></div><div class="hud2" id="hud2"></div><div id="unavail" style="position:absolute;left:12px;right:12px;bottom:12px"></div></div>
  <div class="toolbar" style="margin-top:10px"><button id="pp">▶ Play</button><button id="stepb">Step ▶❘</button><input type="range" class="seek" id="seek" min="0" max="0" value="0" style="flex:1"><span class="mono" id="tcur">0.00 s</span><button id="theatre" title="hide the inspector: wider viewer">▭ Wide</button><button id="fs" title="full screen (Esc to leave)">⛶ Full screen</button></div>
  <div id="note" class="small muted"></div>${renderOk || !health.render.error ? '' : unavailable(health.render.error, 'MuJoCo rendering')}`;
  host.querySelectorAll('[data-mode]').forEach(b => b.onclick = () => { S.mode = b.dataset.mode; S.view = S.mode === 'B' ? 'mujoco:overview' : 'mujoco:overview'; renderSim(host, health, setTop, modes); });
  const appEl = document.getElementById('app'); host.querySelector('#theatre').onclick = () => appEl.classList.toggle('theatre'); host.querySelector('#fs').onclick = () => { const v = host.querySelector('#vwr'); if (document.fullscreenElement) document.exitFullscreen(); else if (v.requestFullscreen) v.requestFullscreen(); };
  const grp = host.querySelector('#grp'), ep = host.querySelector('#ep'), vs = host.querySelector('#viewsel');
  const fillEp = () => { const [src, g] = grp.value.split(':'); ep.innerHTML = S.episodes.filter(e => e.source === src && e.group === g).map(e => `<option value="${esc(e.id)}">${esc(e.title)}${e.outcome ? ' · ' + esc(e.outcome) : ''}</option>`).join(''); };
  const fillViews = () => { const e = S.sel; const mj = ['overview', 'front', 'wrist_left', 'wrist_right', 'free'].map(v => `<option value="mujoco:${v}">MuJoCo · ${v === 'free' ? 'free orbit (drag)' : v}</option>`); const rec = e && e.has_images ? ['front', 'wrist_left', 'wrist_right'].map(v => `<option value="recorded:${v}">Recorded camera · ${v}</option>`) : []; vs.innerHTML = (S.mode === 'B' ? mj : rec.concat(mj)).join(''); if ([...vs.options].some(o => o.value === S.view)) vs.value = S.view; else { S.view = vs.options[0].value; vs.value = S.view; } };
  const open = async id => {
    stop(); S.cache.forEach(u => URL.revokeObjectURL(u)); S.cache.clear(); S.fk = null; S.k = 0; S.err = null; const e = S.episodes.find(x => x.id === id); S.sel = e;
    const [t, m] = await Promise.all([get(`/api/episodes/${encodeURIComponent(id)}/telemetry`), get(`/api/episodes/${encodeURIComponent(id)}/metadata`)]);
    if (!t.ok) { S.tel = null; host.querySelector('#unavail').innerHTML = unavailable(t.reason, 'Episode telemetry'); drawInspector(); drawTimeline(); return; }
    S.tel = t.data; S.meta = m.ok ? m.data : null; S.view = S.sel.has_images ? 'recorded:front' : 'mujoco:overview'; fillViews();   // episodes with stored camera images open instantly on the recorded camera host.querySelector('#seek').max = S.tel.n_frames - 1; host.querySelector('#seek').value = 0;
    host.querySelector('#note').innerHTML = `<b>${esc(S.meta ? S.meta.label : '')}</b>. ${S.tel.limitations.map(esc).join(' · ')}`; setTop({ exp: S.sel.title, src: S.meta ? S.meta.label : S.sel.source, ev: (S.meta && S.meta.source) || '' }); show(0);
  };
  if (S.mode === 'B') {
    S.sel = null; S.tel = null; srcbarB(); S.view = 'mujoco:overview'; fillViews(); host.querySelector('#note').textContent = 'MODE B: the OpenArm v1 lab scene at its home pose, rendered read-only by MuJoCo. Drag in the "free orbit" view to rotate, scroll to zoom. No trajectory, no policy.'; setTop({ exp: 'MuJoCo scene view', src: 'repository lab scene (sim/lab_scene.py)', ev: 'IMPLEMENTED' }); show(0);
  } else if (!S.episodes.length) host.querySelector('#unavail').innerHTML = unavailable('no recorded episodes were found. outputs/ is not stored in git: download the demo bundle and start the server with --data-root <bundle directory> (see app/README.md).', 'Recorded episodes');
  else { fillEp(); grp.onchange = () => { fillEp(); open(ep.value); }; ep.onchange = () => open(ep.value); open(ep.value); }
  function srcbarB() { host.querySelector('#grp').style.display = host.querySelector('#ep').style.display = 'none'; }
  vs.onchange = () => { S.view = vs.value; show(S.k); }; host.querySelector('#spd').onchange = e => { S.speed = +e.target.value; };
  host.querySelector('#pp').onclick = () => S.playing ? stop() : play(); host.querySelector('#stepb').onclick = () => { stop(); if (S.tel) show(Math.min(S.k + 1, S.tel.n_frames - 1)); };
  host.querySelector('#seek').oninput = e => { stop(); show(+e.target.value); };
  const vw = host.querySelector('#vwr'); let drag = null;
  vw.onmousedown = e => { if (S.view.endsWith('free') || S.mode === 'B' && S.view.endsWith('free')) drag = [e.clientX, e.clientY]; }; window.onmouseup = () => drag = null;
  vw.onmousemove = e => { if (!drag) return; S.orbit.az += (e.clientX - drag[0]) * 0.5; S.orbit.el = Math.max(-85, Math.min(-3, S.orbit.el + (e.clientY - drag[1]) * 0.4)); drag = [e.clientX, e.clientY]; clearTimeout(S.ot); S.ot = setTimeout(() => show(S.k), 120); };
  vw.onwheel = e => { if (!S.view.endsWith('free')) return; e.preventDefault(); S.orbit.dist = Math.max(0.8, Math.min(4, S.orbit.dist * (e.deltaY > 0 ? 1.08 : 0.92))); clearTimeout(S.ot); S.ot = setTimeout(() => show(S.k), 120); };
}
function row(k, v) { return `<span>${esc(k)}</span><span>${v}</span>`; }
function drawInspector() {
  const el = document.getElementById('inspector'); if (!el) return; const t = S.tel; if (!t) { el.innerHTML = `<h3>Inspector</h3><p class="muted">${S.mode === 'B' ? 'Scene view: no telemetry (no trajectory loaded).' : 'Select an episode.'}</p>`; return; }
  const k = S.k, st = t.state[k]; const lim = (i) => i === 7 || i === 15 ? [0, 0.05] : [-3.2, 3.2];
  const joints = t.joint_names.map((n, i) => { const [lo, hi] = lim(i), v = st[i], f = Math.max(0, Math.min(1, (v - lo) / (hi - lo))); return `<div class="jrow"><span>${esc(n.replace('openarm_', ''))}</span><span>${fmt(v, 4)} ${i === 7 || i === 15 ? 'm' : 'rad'}</span><div class="jb"><i style="left:0;width:${100 * f}%"></i></div></div>`; }).join('');
  const f = S.fk, tcp = t.tcp_pos ? t.tcp_pos[k] : null;
  const fkHtml = !S.health || S.health.render.error ? '<span class="muted">orientation needs the MuJoCo model (rendering unavailable)</span>' : (f ? (f.error ? `<span class="no">${esc(f.error)}</span>` : `${f.tcp_quat_fk_wxyz.map(x => fmt(x, 3)).join(', ')} <span class="muted">(w,x,y,z, FK)</span><br><span class="muted">|FK − recorded| = ${f.fk_vs_recorded_m !== null ? (f.fk_vs_recorded_m * 1000).toFixed(2) + ' mm' : 'n/a'}</span>`) : '<span class="muted">computing…</span>');
  const c = t.contacts; const cf = c => c ? fmt(c[k], 3) : '<span class="muted">not recorded</span>';
  const ev = t.events.map(e => `<li>t=${e.t} (${(e.t * DT).toFixed(2)} s) <b>${esc(e.kind)}</b> <span class="muted small">${esc(e.note)}</span></li>`).join('');
  el.innerHTML = `<h3>Robot joint state</h3>${joints}<h3>TCP</h3><div class="kv">${row('position [m]', tcp ? tcp.map(x => fmt(x, 4)).join(', ') : 'not recorded')}${row('orientation', fkHtml)}</div>
  <h3>Gripper</h3><div class="kv">${row('left opening [m]', fmt(t.gripper_left_m[k], 4))}${row('right opening [m]', fmt(t.gripper_right_m[k], 4))}${row('left command [m]', t.action && k < t.n_actions ? fmt(t.action[k][7], 4) : 'not recorded')}</div>
  <h3>Contact</h3><div class="kv">${row('finger force [N]', c ? cf(c.finger_force) : '<span class="muted">not recorded</span>')}${row('finger contacts', c ? cf(c.finger_contacts) : '—')}${row('palm force [N]', c ? cf(c.hand_force) : '—')}${c && c.table_force ? row('table force [N]', cf(c.table_force)) : ''}</div>
  <h3>Box</h3><div class="kv">${row('position [m]', t.box_pos ? t.box_pos[k].map(x => fmt(x, 4)).join(', ') : 'not recorded')}${row('orientation', t.box_quat ? t.box_quat[k].map(x => fmt(x, 3)).join(', ') + ' (w,x,y,z)' : t.box_yaw ? 'yaw ' + fmt(t.box_yaw[k], 3) + ' rad' : '<span class="muted">not recorded</span>')}</div>
  <h3>Derived events</h3><ul class="small">${ev || '<li class="muted">none found</li>'}</ul><p class="small muted">${esc(t.events_note)}</p>
  <h3>Recorded outcome</h3><div class="kv">${t.outcome ? Object.entries(t.outcome).map(([a, b]) => row(a, esc(JSON.stringify(b)))).join('') : row('outcome', 'not recorded')}</div>
  <h3>Model predictions</h3><p class="small muted">${esc(t.model_predictions_note)}</p>`;
}
function drawTimeline() {
  const cv = document.getElementById('tl'); if (!cv) return; const t = S.tel; const dpr = window.devicePixelRatio || 1; const W = cv.clientWidth, Hh = 120; cv.width = W * dpr; cv.height = Hh * dpr; const g = cv.getContext('2d'); g.scale(dpr, dpr); g.clearRect(0, 0, W, Hh);
  g.fillStyle = '#7b8fb0'; g.font = '10px sans-serif';
  if (!t) { g.fillText(S.mode === 'B' ? 'Scene view: no timeline' : 'No episode loaded', 10, 20); return; }
  const n = t.n_frames, x = i => 50 + (W - 70) * i / Math.max(1, n - 1);
  const series = (arr, lo, hi, col, label, row) => { if (!arr) return; g.strokeStyle = col; g.lineWidth = 1.4; g.beginPath(); arr.slice(0, n).forEach((v, i) => { const y = 100 - 80 * Math.max(0, Math.min(1, (v - lo) / (hi - lo))); i ? g.lineTo(x(i), y) : g.moveTo(x(i), y); }); g.stroke(); g.fillStyle = col; g.fillText(label, 6, 14 + row * 11); };
  series(t.gripper_left_m, 0, 0.05, '#41d8ea', 'gripper', 0); if (t.action) series(t.action.map(a => a[7]), 0, 0.05, '#2bb8a3', 'cmd', 1);
  if (t.box_pos) { const z0 = t.box_pos[0][2]; series(t.box_pos.map(p => p[2] - z0), 0, 0.12, '#f0b45a', 'box rise', 2); }
  if (t.contacts && t.contacts.finger_force) series(t.contacts.finger_force, 0, Math.max(10, ...t.contacts.finger_force.filter(Number.isFinite)), '#ef6b73', 'finger N', 3);
  if (t.phases) { const pc = ['#12263f', '#102a2a', '#2a2210', '#231a38', '#10283a']; t.phases.forEach((p, i) => { g.fillStyle = pc[i % pc.length]; g.fillRect(x(p.start), 2, Math.max(1, x(p.end) - x(p.start)), 10); g.fillStyle = '#7b8fb0'; g.font = '9px sans-serif'; if (x(p.end) - x(p.start) > 38) g.fillText(p.name, x(p.start) + 2, 10); }); g.font = '10px sans-serif'; }
  (t.chunk_boundaries || []).forEach(i => { g.strokeStyle = '#1f3558'; g.beginPath(); g.moveTo(x(i), 8); g.lineTo(x(i), 104); g.stroke(); });
  const colr = { close_command: '#2bb8a3', lift: '#f0b45a', release: '#9d8cf5' };
  t.events.forEach(e => { g.strokeStyle = colr[e.kind] || '#fff'; g.setLineDash([3, 3]); g.beginPath(); g.moveTo(x(e.t), 6); g.lineTo(x(e.t), 104); g.stroke(); g.setLineDash([]); g.fillStyle = colr[e.kind] || '#fff'; g.fillText(e.kind, x(e.t) + 3, 112); });
  g.fillStyle = '#7b8fb0'; g.fillText((t.chunk_boundaries || []).length ? 'action-chunk boundaries every 20 steps (thin lines)' : 'no action chunks in this source', W - 270, 14);
  g.strokeStyle = '#ffffff'; g.lineWidth = 1.5; g.beginPath(); g.moveTo(x(S.k), 4); g.lineTo(x(S.k), 106); g.stroke();
  cv.onclick = e => { const r = cv.getBoundingClientRect(); const i = Math.round((e.clientX - r.left - 50) / (W - 70) * (n - 1)); stop(); show(Math.max(0, Math.min(n - 1, i))); };
}
