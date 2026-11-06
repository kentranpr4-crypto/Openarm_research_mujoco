// Interactive node-and-edge architecture renderer (plain SVG).  Layers are never blended: only the selected layer is drawn and each node lists all layers it belongs to.
import { esc, badge, unavailable } from './ui.js';
const W = 220, H = 70;
export function renderGraph(svgHost, nodes, edges, onSelect, selected) {
  const by = Object.fromEntries(nodes.map(n => [n.id, n]));
  const maxX = Math.max(...nodes.map(n => n.x)) + W + 40, maxY = Math.max(...nodes.map(n => n.y)) + H + 40;
  const path = e => {
    const a = by[e.s], b = by[e.t]; if (!a || !b) return '';
    const ax = a.x + W / 2, ay = a.y + H / 2, bx = b.x + W / 2, by_ = b.y + H / 2; let x1 = ax, y1 = ay, x2 = bx, y2 = by_;
    if (Math.abs(bx - ax) > Math.abs(by_ - ay)) { x1 = ax + Math.sign(bx - ax) * W / 2; x2 = bx - Math.sign(bx - ax) * W / 2; } else { y1 = ay + Math.sign(by_ - ay) * H / 2; y2 = by_ - Math.sign(by_ - ay) * H / 2; }
    return `<path class="edge ${e.label === 'feedback' || /replan/.test(e.label || '') ? 'fb' : ''}" d="M${x1},${y1} L${x2},${y2}" marker-end="url(#ar)"/>${e.label ? `<text class="elabel" x="${(x1 + x2) / 2 + 4}" y="${(y1 + y2) / 2 - 4}">${esc(e.label)}</text>` : ''}`;
  };
  const nd = n => `<g class="node ${(n.status || '').replace(/[^A-Z]/g, '')} ${selected === n.id ? 'sel' : ''}" data-id="${esc(n.id)}" transform="translate(${n.x},${n.y})" style="cursor:pointer"><rect width="${W}" height="${H}" rx="9"/><text x="12" y="27">${esc(n.label.length > 34 ? n.label.slice(0, 33) + '…' : n.label)}</text><text class="sub" x="12" y="47">${esc(n.status || '')}</text><text class="sub" x="12" y="61">${esc((n.layers || []).join(' · '))}</text></g>`;
  svgHost.innerHTML = `<svg viewBox="0 0 ${maxX} ${maxY}" width="100%" style="max-height:62vh"><defs><marker id="ar" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 z" fill="#3a5f94"/></marker></defs>${edges.map(path).join('')}${nodes.map(nd).join('')}</svg>`;
  svgHost.querySelectorAll('.node').forEach(g => g.addEventListener('click', () => onSelect(g.dataset.id)));
}
export function nodePanel(n) {
  if (!n) return '<p class="muted">Click a node to inspect its implementation, inputs, outputs and limitations.</p>';
  const list = a => (a && a.length) ? `<ul>${a.map(x => `<li class="mono small">${esc(x)}</li>`).join('')}</ul>` : '<p class="muted">—</p>';
  const files = (n.files_status || []).map(f => `<li class="mono small ${f.exists ? 'ok' : 'no'}">${f.exists ? '✓' : '✗'} ${esc(f.path)}</li>`).join('');
  return `<div class="card"><div>${badge(n.status === 'IMPLEMENTED' ? 'IMPLEMENTED' : n.status || 'DESIGN ONLY')}</div><h3>${esc(n.label)}</h3><p>${esc(n.purpose)}</p>
  <h3>Implementation files</h3><ul>${files || '<li class="muted">none (not implemented)</li>'}</ul><h3>Input data</h3>${list(n.inputs || (n.proposed_io || {}).inputs)}<h3>Output data</h3>${list(n.outputs || (n.proposed_io || {}).outputs)}
  <h3>Tensor shapes</h3><p class="small muted">${esc(n.shapes_verified ? 'Verified by: ' + n.shapes_verified : 'PROPOSED - not implemented, not verified')}</p>
  <h3>Model type</h3><p>${esc(n.model_type || 'proposed')}</p><h3>Training / inference</h3><p>${esc(n.status || '')}${(n.artifacts || []).length ? '<br><span class="small mono">' + n.artifacts.map(esc).join('<br>') + '</span>' : ''}</p>
  <h3>Known limitations</h3>${list(n.limitations || (n.sim2real ? ['Sim2Real: ' + n.sim2real] : []))}<h3>Relevant experiment</h3><p>${esc(n.experiment || '—')}</p><h3>Provenance</h3><p class="small muted">${esc(n.provenance || 'design document')}</p></div>`;
}
export { unavailable };
