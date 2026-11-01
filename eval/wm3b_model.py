#!/usr/bin/env python3
"""WM3-B: ridge joint-trajectory forecast + GRU object/contact model (frozen spec: outputs/wm3/model_config_wm3b.json).
  * Ridge (closed form, alpha 10, the SAME estimator as WM-2's ridge, fitted on train windows only) forecasts the future left-arm joint deltas Y_s (20,8) from the INPUT blocks only.
  * GRU-OC consumes the history, the candidate executed-action sequence AND the (normalised) ridge joint forecast at every decoder step; it predicts box (7) and contact (7) per step and the 4 event logits.
  * No joint head / joint loss in the GRU; loss = MSE(Y_b) + 0.5*MSE(Y_c) + 0.5*BCE(Y_e).  Ground-truth future joints are never an input (train uses the ridge forecast, exactly as inference does)."""
import sys, time, json, numpy as np, torch, torch.nn as nn
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "eval"))
import wm2_models as M
K, H = 20, 4
class GRUOC(nn.Module):
    def __init__(self):
        super().__init__(); self.enc = nn.Sequential(nn.Linear(H * 30, 256), nn.ReLU()); self.init = nn.Linear(256, 2 * 256); self.ain = nn.Linear(8, 64); self.sin = nn.Linear(8, 32); self.gru = nn.GRU(96, 256, 2, batch_first=True); self.out = nn.Linear(256, 14); self.ev = nn.Linear(256, 4)
    def forward(self, hist, afut, rfc):
        c = self.enc(hist); h0 = self.init(c).view(-1, 2, 256).transpose(0, 1).contiguous(); x = torch.cat([torch.relu(self.ain(afut)), torch.relu(self.sin(rfc))], -1); o, _ = self.gru(x, h0); y = self.out(o)
        return torch.cat([y[:, :, :7].reshape(len(y), -1), y[:, :, 7:].reshape(len(y), -1)], 1), self.ev(o.mean(1))
def loss_oc(pred, y, ev, logit):
    f = nn.functional.mse_loss; return f(pred[:, :140], y[:, :140]) + 0.5 * f(pred[:, 140:], y[:, 140:]) + 0.5 * nn.functional.binary_cross_entropy_with_logits(logit, ev)
def ridge_forecast(ridge, nrm, W):
    """The ONLY path producing the joint forecast (train and inference): raw deltas (N,20,8) and their normalised version.  Uses input blocks only."""
    raw = ridge.predict(W)["Y_s"]; return raw, nrm.n(raw, nrm.Ys)
class WM3B:
    def __init__(self, ridge, nrm): self.ridge, self.nrm = ridge, nrm
    def tensors(self, W, rfc_override=None):
        raw, rn = ridge_forecast(self.ridge, self.nrm, W)
        if rfc_override is not None: rn = rfc_override
        return torch.tensor(self.nrm.hist(W)), torch.tensor(self.nrm.n(W["A_fut"], self.nrm.A)), torch.tensor(rn.astype(np.float32)), raw
    def targets(self, W): n = len(W["Y_b"]); return torch.tensor(np.concatenate([self.nrm.n(W["Y_b"], self.nrm.Yb).reshape(n, -1), self.nrm.n(W["Y_c"], self.nrm.Yc).reshape(n, -1)], 1))
    def fit(self, Wtr, Wva, log, seed=0, epochs=20, max_steps=None, lr=1e-3, wd=1e-4, bs=128):
        torch.manual_seed(seed); np.random.seed(seed); self.model = GRUOC(); self.n_params = sum(p.numel() for p in self.model.parameters()); Xtr, Xva = self.tensors(Wtr)[:3], self.tensors(Wva)[:3]; Ytr, Yva = self.targets(Wtr), self.targets(Wva); Etr, Eva = torch.tensor(Wtr["Y_e"]), torch.tensor(Wva["Y_e"])
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd); n = len(Ytr); steps = epochs * (n // bs); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, 1e-5); best = (1e9, None, 0); t0 = time.time(); step = 0; g = torch.Generator().manual_seed(seed)
        for ep in range(epochs):
            self.model.train(); perm = torch.randperm(n, generator=g); tl = 0.0; nb = 0
            for i in range(0, n - bs + 1, bs):
                idx = perm[i:i + bs]; pr, lg = self.model(*(x[idx] for x in Xtr)); loss = loss_oc(pr, Ytr[idx], Etr[idx], lg); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0); opt.step(); sched.step(); step += 1; tl += loss.item(); nb += 1
                if max_steps and step >= max_steps: break
            self.model.eval()
            with torch.no_grad(): vl = 0.0; vb = 0; [(lambda p, l: None) for _ in []]; vl = sum(loss_oc(*(lambda o: (o[0], Yva[i:i + 2048], Eva[i:i + 2048], o[1]))(self.model(*(x[i:i + 2048] for x in Xva)))).item() for i in range(0, len(Yva), 2048)); vb = len(range(0, len(Yva), 2048))
            vl /= vb; log.append(dict(model="wm3b", epoch=ep + 1, train_loss=tl / max(nb, 1), val_loss=vl, elapsed_s=time.time() - t0, finite=bool(np.isfinite(tl))))
            if vl < best[0]: best = (vl, {k: v.clone() for k, v in self.model.state_dict().items()}, ep + 1)
            if max_steps and step >= max_steps: break
        self.model.load_state_dict(best[1]); self.model.eval(); self.best_epoch = best[2]; self.train_seconds = time.time() - t0; return self
    def predict(self, W, rfc_override=None):
        self.model.eval(); hist, af, rn, raw = self.tensors(W, rfc_override); outs, evs = [], []
        with torch.no_grad():
            for i in range(0, len(hist), 4096): p, lg = self.model(hist[i:i + 4096], af[i:i + 4096], rn[i:i + 4096]); outs.append(p.numpy()); evs.append(torch.sigmoid(lg).numpy())
        y = np.concatenate(outs); n = len(y); return dict(Y_s=raw.astype(np.float32), Y_b=(y[:, :140].reshape(n, K, 7) * self.nrm.Yb[1] + self.nrm.Yb[0]).astype(np.float32), Y_c=(y[:, 140:].reshape(n, K, 7) * self.nrm.Yc[1] + self.nrm.Yc[0]).astype(np.float32), Y_e=np.concatenate(evs).astype(np.float32))
