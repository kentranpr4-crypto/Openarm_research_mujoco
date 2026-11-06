// Thin read-only API client.  Every call resolves to {ok, data} or {ok:false, error, reason}; the UI never invents data on failure.
export async function get(path) {
  try {
    const r = await fetch(path, { headers: { Accept: 'application/json' } });
    const ct = r.headers.get('content-type') || '';
    if (!ct.includes('json')) return r.ok ? { ok: true, data: null } : { ok: false, error: 'http_' + r.status, reason: r.statusText };
    const j = await r.json();
    return r.ok ? { ok: true, data: j } : { ok: false, error: j.error || 'error', reason: j.reason || r.statusText };
  } catch (e) { return { ok: false, error: 'network', reason: String(e) }; }
}
export const frameUrl = (id, k, view, o = {}) => `/api/episodes/${encodeURIComponent(id)}/frame/${k}?view=${encodeURIComponent(view)}&az=${o.az ?? 140}&el=${o.el ?? -25}&dist=${o.dist ?? 1.9}`;
export const sceneUrl = (view, o = {}) => `/api/scene/frame?view=${encodeURIComponent(view)}&az=${o.az ?? 140}&el=${o.el ?? -25}&dist=${o.dist ?? 1.9}`;
