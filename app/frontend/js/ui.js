// Small rendering helpers.  All dynamic text goes through esc().
export const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const BC = { 'IMPLEMENTED': 'IMPLEMENTED', 'VALIDATED': 'VALIDATED', 'EXPERIMENTAL': 'EXPERIMENTAL', 'DESIGN ONLY': 'DESIGN', 'NOT TRAINED': 'NOTTRAINED', 'NOT EVALUATED': 'NOTEVAL' };
export const badge = l => `<span class="badge b-${BC[l] || 'NOTEVAL'}">${esc(l)}</span>`;
export const badges = ls => (ls || []).map(badge).join('');
export const validityBadge = v => { const bad = /INVALID/.test(v || ''); return `<span class="badge ${bad ? 'b-INVALID' : 'b-NOTEVAL'}">${esc(String(v || '').replace(/_/g, ' '))}</span>`; };
export const unavailable = (reason, what = 'Data') => `<div class="unavail"><b>${esc(what)} unavailable.</b> ${esc(reason || 'no reason reported')}</div>`;
export const fmt = (v, d = 3) => (typeof v === 'number' && isFinite(v)) ? (Math.abs(v) >= 1000 || (Math.abs(v) > 0 && Math.abs(v) < 0.001) ? v.toExponential(2) : v.toFixed(d).replace(/\.?0+$/, '') || '0') : (v === null || v === undefined ? '—' : esc(v));
export const pct = v => typeof v === 'number' ? (100 * v).toFixed(1).replace(/\.0$/, '') + ' %' : '—';
function kn(v) {   // {k,n,rate,wilson95}
  const [lo, hi] = v.wilson95 || [0, 0];
  return `<div class="mv">${v.k}/${v.n} <span class="muted">(${pct(v.rate)})</span></div><div class="small muted">Wilson 95 % [${pct(lo)} – ${pct(hi)}]</div><div class="bar"><i style="left:${100 * lo}%;width:${100 * (hi - lo)}%"></i><b style="left:calc(${100 * v.rate}% - 1px)"></b></div>`;
}
function plain(v) {
  if (v === null || v === undefined) return '<span class="muted">—</span>';
  if (typeof v !== 'object') return `<div class="mv">${fmt(v)}</div>`;
  if (v.k !== undefined && v.n !== undefined && v.wilson95) return kn(v);
  if (Array.isArray(v) && v.length && typeof v[0] === 'object' && v[0] !== null) { const cols = Object.keys(v[0]); return `<table>${'<tr>' + cols.map(c => `<th>${esc(c)}</th>`).join('') + '</tr>'}${v.map(r => '<tr>' + cols.map(c => `<td class="small">${typeof r[c] === 'object' ? esc(JSON.stringify(r[c])) : fmt(r[c])}</td>`).join('') + '</tr>').join('')}</table>`; }
  if (Array.isArray(v)) return `<div class="mv">[${v.map(x => fmt(x)).join(', ')}]</div>`;
  return `<div class="kv">${Object.entries(v).map(([k, x]) => `<span>${esc(k)}</span><span>${typeof x === 'object' ? esc(JSON.stringify(x)) : fmt(x)}</span>`).join('')}</div>`;
}
export function metric(m) {
  const tag = m.tag ? `<span class="tag">${esc(m.tag)}</span>` : '';
  const src = m.source ? `<div class="small muted mono">source: ${esc(m.source)}${m.path ? ' › ' + esc(m.path) : ''}</div>` : '';
  const doc = m.doc_check ? `<div class="small ${m.doc_check.found ? 'ok' : 'no'}">${m.doc_check.found ? '✓' : '✗'} report text ${m.doc_check.found ? 'confirms' : 'does not contain'} "${esc(m.doc_check.contains)}"</div>` : '';
  const body = m.status === 'UNAVAILABLE' || m.status === 'NOT_EVALUATED' ? `<div class="unavail">${esc(m.status)}: ${esc(m.reason || '')}</div>` : plain(m.value);
  return `<div class="metric"><div class="ml">${esc(m.label)}${m.unit ? ` <span class="muted">[${esc(m.unit)}]</span>` : ''} ${tag}</div>${body}${m.note ? `<div class="small muted">${esc(m.note)}</div>` : ''}${doc}${src}</div>`;
}
export function experimentDetail(c) {
  const arts = (c.artifacts || []).map(a => `<tr><td class="mono small">${esc(a.path)}</td><td>${esc(a.role)}</td><td class="${a.exists ? 'ok' : 'no'}">${a.exists ? 'present' : 'missing'}</td><td class="mono small">${a.sha256 ? esc(a.sha256.slice(0, 16)) + '…' : '—'}</td></tr>`).join('');
  const fm = (c.failure_modes || []).map(f => `<li>${typeof f === 'string' ? esc(f) : esc(f.mode) + ' <span class="muted">(' + esc(f.count) + ')</span>'}</li>`).join('');
  return `<div class="card"><div>${badges(c.labels)}${validityBadge(c.outcome_validity.status)}</div><h1>${esc(c.title)}</h1><p class="lead">${esc(c.purpose)}</p>
  <div class="grid g2"><div><h3>Dataset / poses</h3><p>${esc(c.dataset)}</p><h3>Candidates / policies</h3><ul>${(c.candidates || []).map(x => `<li>${esc(x)}</li>`).join('')}</ul><h3>Evaluation method</h3><p>${esc(c.method)}</p></div>
  <div><h3>Outcome validity</h3><div class="banner ${/INVALID|NOT_EVALUATED/.test(c.outcome_validity.status) ? 'bad' : 'warn'}"><b>${esc(String(c.outcome_validity.status).replace(/_/g, ' '))}</b> — ${esc(c.outcome_validity.text)}</div><h3>Known failure modes</h3><ul>${fm || '<li class="muted">none recorded</li>'}</ul><h3>Caveats</h3><ul>${(c.caveats || []).map(x => `<li>${esc(x)}</li>`).join('')}</ul></div></div>
  <h2>Metrics (read from saved artifacts)</h2>${c.metrics.map(metric).join('')}
  <h2>Provenance</h2><table><tr><th>Artifact</th><th>Role</th><th>State</th><th>sha256</th></tr>${arts}</table>${c.freeze ? `<p class="small muted">Freeze file: <span class="mono">${esc(c.freeze.file)}</span> sha256 <span class="mono">${esc(c.freeze.sha256 || 'n/a')}</span></p>` : ''}</div>`;
}
