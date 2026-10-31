#!/usr/bin/env python3
"""WM-2 models (frozen configs in outputs/wm2/model_config.json): ridge, MLP (+ action-blind variants), outcome-only logistic, and the action-conditioned GRU prototype.  CPU only, one training seed (0).
All models map normalised windows to per-step deltas Y_s (20,8), Y_b (20,7), contact block Y_c (20,7) and event probabilities Y_e (4); predictions are returned in RAW units."""
import time, json, numpy as np, torch, torch.nn as nn
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; CFG = json.loads((ROOT / "outputs/wm2/model_config.json").read_text())
K, H = 20, 4
class Norm:
    """Train-only statistics."""
    def __init__(self, W):
        f = lambda x: (x.reshape(-1, x.shape[-1]).mean(0), x.reshape(-1, x.shape[-1]).std(0) + 1e-6)
        self.S, self.B, self.C = f(W["S_hist"]), f(W["B_hist"]), f(W["C_hist"]); self.A = f(np.concatenate([W["A_hist"], W["A_fut"]], 1)); self.Ys, self.Yb, self.Yc = f(W["Y_s"]), f(W["Y_b"]), f(W["Y_c"]); self.pose = f(W["pose"])
    def n(self, x, st): return ((x - st[0]) / st[1]).astype(np.float32)
    def hist(self, W, blind=False):
        parts = [self.n(W["S_hist"], self.S), self.n(W["B_hist"], self.B), self.n(W["C_hist"], self.C)]
        if not blind: parts.append(self.n(W["A_hist"], self.A))
        return np.concatenate([p.reshape(len(p), -1) for p in parts], 1)
    def feat(self, W, kind):
        if kind == "outcome_only": return np.concatenate([self.n(W["S_hist"][:, -1], self.S), self.n(W["B_hist"][:, -1], self.B), self.n(W["C_hist"][:, -1], self.C), self.n(W["pose"], self.pose)], 1).astype(np.float32)
        h = self.hist(W, blind=(kind == "blind"))
        return h if kind == "blind" else np.concatenate([h, self.n(W["A_fut"], self.A).reshape(len(h), -1)], 1)
    def targets(self, W): n = len(W["Y_s"]); return np.concatenate([self.n(W["Y_s"], self.Ys).reshape(n, -1), self.n(W["Y_b"], self.Yb).reshape(n, -1), self.n(W["Y_c"], self.Yc).reshape(n, -1)], 1)
    def untarget(self, Y):
        n = len(Y); a = Y[:, :K * 8].reshape(n, K, 8); b = Y[:, K * 8:K * 15].reshape(n, K, 7); c = Y[:, K * 15:].reshape(n, K, 7)
        return dict(Y_s=(a * self.Ys[1] + self.Ys[0]).astype(np.float32), Y_b=(b * self.Yb[1] + self.Yb[0]).astype(np.float32), Y_c=(c * self.Yc[1] + self.Yc[0]).astype(np.float32))
def logistic_fit(X, y, l2=0.01, iters=200):
    X = torch.tensor(X, dtype=torch.float32); y = torch.tensor(y, dtype=torch.float32); w = torch.zeros(X.shape[1], y.shape[1], requires_grad=True); b = torch.zeros(y.shape[1], requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=iters, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad(); loss = nn.functional.binary_cross_entropy_with_logits(X @ w + b, y) + 0.5 * l2 * (w ** 2).sum(); loss.backward(); return loss
    opt.step(closure); return w.detach().numpy(), b.detach().numpy()
def sigmoid(z): return 1 / (1 + np.exp(-z))
class Ridge:
    def __init__(self, kind): self.kind = kind
    def fit(self, nrm, W):
        self.nrm = nrm; X = nrm.feat(W, self.kind).astype(np.float64); self.mu = X.mean(0); self.sd = X.std(0) + 1e-6; X = (X - self.mu) / self.sd; X1 = np.concatenate([X, np.ones((len(X), 1))], 1); Y = nrm.targets(W).astype(np.float64); a = CFG["ridge"]["alpha"]
        R = np.eye(X1.shape[1]) * a; R[-1, -1] = 0; self.Wt = np.linalg.solve(X1.T @ X1 + R, X1.T @ Y); self.lw, self.lb = logistic_fit(X.astype(np.float32), W["Y_e"], CFG["ridge"]["events"] and 0.01, 200); return self
    def predict(self, W):
        X = (self.nrm.feat(W, self.kind).astype(np.float64) - self.mu) / self.sd; X1 = np.concatenate([X, np.ones((len(X), 1))], 1); out = self.nrm.untarget(X1 @ self.Wt); out["Y_e"] = sigmoid(X.astype(np.float32) @ self.lw + self.lb).astype(np.float32); return out
class OutcomeOnly:
    def fit(self, nrm, W):
        self.nrm = nrm; X = nrm.feat(W, "outcome_only"); self.mu = X.mean(0); self.sd = X.std(0) + 1e-6; self.lw, self.lb = logistic_fit((X - self.mu) / self.sd, W["Y_e"], 0.01, 200); return self
    def predict(self, W): X = (self.nrm.feat(W, "outcome_only") - self.mu) / self.sd; return dict(Y_e=sigmoid(X @ self.lw + self.lb).astype(np.float32))
def loss_fn(pred, y, ev, logit):
    n = pred.shape[0]; l = lambda a, b: nn.functional.mse_loss(a, b)
    return l(pred[:, :K * 8], y[:, :K * 8]) + l(pred[:, K * 8:K * 15], y[:, K * 8:K * 15]) + 0.5 * l(pred[:, K * 15:], y[:, K * 15:]) + 0.5 * nn.functional.binary_cross_entropy_with_logits(logit, ev)
class MLP(nn.Module):
    def __init__(self, d_in):
        super().__init__(); h = CFG["mlp"]["hidden"]; self.net = nn.Sequential(nn.Linear(d_in, h[0]), nn.ReLU(), nn.Dropout(CFG["mlp"]["dropout"]), nn.Linear(h[0], h[1]), nn.ReLU(), nn.Dropout(CFG["mlp"]["dropout"]), nn.Linear(h[1], K * 22 + 4))
    def forward(self, x): o = self.net(x); return o[:, :K * 22], o[:, K * 22:]
class GRUModel(nn.Module):
    def __init__(self):
        super().__init__(); c = CFG["gru"]; self.enc = nn.Sequential(nn.Linear(H * 30, 256), nn.ReLU()); self.init = nn.Linear(256, 2 * 256); self.ain = nn.Linear(8, 64); self.gru = nn.GRU(64, 256, 2, batch_first=True); self.out = nn.Linear(256, 22); self.ev = nn.Linear(256, 4)
    def forward(self, hist, afut):
        c = self.enc(hist); h0 = self.init(c).view(-1, 2, 256).transpose(0, 1).contiguous(); o, _ = self.gru(torch.relu(self.ain(afut)), h0); y = self.out(o)                                  # (B,K,22)
        return torch.cat([y[:, :, :8].reshape(len(y), -1), y[:, :, 8:15].reshape(len(y), -1), y[:, :, 15:].reshape(len(y), -1)], 1), self.ev(o.mean(1))
class NN:
    """Gradient-trained model wrapper (MLP / GRU) with best-validation selection."""
    def __init__(self, kind): self.kind = kind          # 'mlp_cond' | 'mlp_blind' | 'gru'
    def _inp(self, W):
        if self.kind == "gru": return torch.tensor(self.nrm.hist(W)), torch.tensor(self.nrm.n(W["A_fut"], self.nrm.A))
        return (torch.tensor(self.nrm.feat(W, "blind" if self.kind == "mlp_blind" else "cond")),)
    def _fwd(self, inp): return self.model(*inp)
    def fit(self, nrm, Wtr, Wva, log, seed=0, epochs=None, max_steps=None):
        torch.manual_seed(seed); np.random.seed(seed); torch.set_num_threads(CFG["gru" if self.kind == "gru" else "mlp"]["threads"]); self.nrm = nrm; c = CFG["gru" if self.kind == "gru" else "mlp"]
        self.model = GRUModel() if self.kind == "gru" else MLP(self._inp(Wtr)[0].shape[1]); self.n_params = sum(p.numel() for p in self.model.parameters()); Xtr, Xva = self._inp(Wtr), self._inp(Wva); Ytr, Yva = torch.tensor(nrm.targets(Wtr)), torch.tensor(nrm.targets(Wva)); Etr, Eva = torch.tensor(Wtr["Y_e"]), torch.tensor(Wva["Y_e"])
        opt = torch.optim.AdamW(self.model.parameters(), lr=c["lr"], weight_decay=c["weight_decay"]); E = epochs or c["epochs"]; bs = c["batch"]; n = len(Ytr); steps = E * (n // bs); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, 1e-5) if self.kind == "gru" else None
        best = (1e9, None); t0 = time.time(); step = 0; g = torch.Generator().manual_seed(seed)
        for ep in range(E):
            self.model.train(); perm = torch.randperm(n, generator=g); tl = 0.0; nb = 0
            for i in range(0, n - bs + 1, bs):
                idx = perm[i:i + bs]; pred, lg = self._fwd(tuple(x[idx] for x in Xtr)); loss = loss_fn(pred, Ytr[idx], Etr[idx], lg); opt.zero_grad(); loss.backward()
                if self.kind == "gru": torch.nn.utils.clip_grad_norm_(self.model.parameters(), CFG["gru"]["grad_clip"])
                opt.step(); step += 1; tl += loss.item(); nb += 1
                if sched: sched.step()
                if max_steps and step >= max_steps: break
            self.model.eval()
            with torch.no_grad():
                vl = 0.0; vb = 0
                for i in range(0, len(Yva), 2048): p, lg = self._fwd(tuple(x[i:i + 2048] for x in Xva)); vl += loss_fn(p, Yva[i:i + 2048], Eva[i:i + 2048], lg).item(); vb += 1
            vl /= vb; log.append(dict(model=self.kind, epoch=ep + 1, train_loss=tl / max(nb, 1), val_loss=vl, elapsed_s=time.time() - t0, finite=bool(np.isfinite(tl))))
            if vl < best[0]: best = (vl, {k: v.clone() for k, v in self.model.state_dict().items()}, ep + 1)
            if max_steps and step >= max_steps: break
        self.model.load_state_dict(best[1]); self.model.eval(); self.best_epoch = best[2]; self.train_seconds = time.time() - t0; return self
    def predict(self, W):
        self.model.eval(); X = self._inp(W); outs = []; evs = []
        with torch.no_grad():
            for i in range(0, len(W["t"] if "t" in W else W["Y_e"]), 4096): p, lg = self._fwd(tuple(x[i:i + 4096] for x in X)); outs.append(p.numpy()); evs.append(torch.sigmoid(lg).numpy())
        out = self.nrm.untarget(np.concatenate(outs)); out["Y_e"] = np.concatenate(evs).astype(np.float32); return out
def latency(model_fn, make_input, n_warm=100, n_run=1000):
    torch.set_num_threads(1); x = make_input()
    with torch.no_grad():
        for _ in range(n_warm): model_fn(x)
        ts = []
        for _ in range(n_run): t = time.perf_counter(); model_fn(x); ts.append((time.perf_counter() - t) * 1000)
    ts = np.array(ts); return dict(median_ms=float(np.median(ts)), p95_ms=float(np.percentile(ts, 95)), p99_ms=float(np.percentile(ts, 99)), n=n_run)
