#!/usr/bin/env python3
import argparse, json, numpy as np, torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn.functional import smooth_l1_loss, l1_loss
from pathlib import Path
from dataset import PolyMAXDataset
from model_predictor import PredictorMLP
from surrogate_mlp import SurrogateMLP

def collate(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs), torch.stack(ys)

def main(manifest, features_dir, schema_json, surrogate_ckpt, sens_weights_json, out_dir,
         lambda_cycle=0.5, lr=2e-3):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    tr = PolyMAXDataset(manifest, features_dir, "train", schema_json)
    va = PolyMAXDataset(manifest, features_dir, "val",   schema_json)
    tl = DataLoader(tr, batch_size=128, shuffle=True, collate_fn=collate)
    vl = DataLoader(va, batch_size=128, shuffle=False, collate_fn=collate)
    in_dim = tr[0][0].numel()

    # Predictor
    model = PredictorMLP(in_dim=in_dim, out_params=66).to(device)

    # Frozen surrogate (params -> features) for cycle/perceptual loss
    s_ck = torch.load(surrogate_ckpt, map_location="cpu")
    surrogate = SurrogateMLP(in_dim=66, out_dim=s_ck["out_dim"]).to(device)
    surrogate.load_state_dict(s_ck["model"]); surrogate.eval()
    for p in surrogate.parameters(): p.requires_grad_(False)

    # Sensitivity weights (optional)
    if sens_weights_json and Path(sens_weights_json).exists():
        W = torch.tensor(json.load(open(sens_weights_json)), dtype=torch.float32, device=device)  # shape (66,)
    else:
        W = torch.ones(66, dtype=torch.float32, device=device)

    opt = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    best = 1e9
    for ep in range(300):
        model.train(); tr_loss = 0.0
        for x,y in tl:
            x,y = x.to(device), y.to(device)
            opt.zero_grad()
            y_hat, _ = model(x)                         # (B,66) in [0,1]
            # Param loss with sensitivity weighting
            Lp = smooth_l1_loss(y_hat, y, reduction="none")   # (B,66)
            Lp = (Lp * W).mean()

            # Cycle/perceptual loss in feature space via surrogate
            f_hat = surrogate(y_hat)                    # (B,F)
            # Build matching audio features x_feat depending on your extractor:
            # - if x are global stats, use x directly; else pass through a small "pool" to match surrogate output dim.
            x_feat = x
            Lc = l1_loss(f_hat, x_feat)

            loss = Lp + lambda_cycle * Lc
            loss.backward(); opt.step()
            tr_loss += loss.item() * x.size(0)
        tr_loss /= len(tr)

        # Validation
        model.eval(); va_loss = 0.0
        with torch.no_grad():
            for x,y in vl:
                x,y = x.to(device), y.to(device)
                y_hat,_ = model(x)
                Lp = smooth_l1_loss(y_hat, y, reduction="none")
                Lp = (Lp * W).mean()
                f_hat = surrogate(y_hat)
                x_feat = x
                Lc = l1_loss(f_hat, x_feat)
                va_loss += (Lp + lambda_cycle*Lc).item() * x.size(0)
        va_loss /= len(va)
        print(f"[predictor] ep{ep:03d} train {tr_loss:.4f} val {va_loss:.4f}")

        if va_loss < best - 1e-4:
            best = va_loss
            torch.save({"model": model.state_dict(), "in_dim": in_dim}, out/"predictor_best.pt")
        if ep>10 and va_loss>best+0.02: break

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--features_dir", default="data/features")
    ap.add_argument("--schema_json", default="data/params_schema.json")
    ap.add_argument("--surrogate_ckpt", default="checkpoints/surrogate.pt")
    ap.add_argument("--sens_weights_json", default="data/sensitivity_weights.json")
    ap.add_argument("--out_dir", default="checkpoints/predictor")
    ap.add_argument("--lambda_cycle", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=2e-3)
    args = ap.parse_args()
    main(**vars(args))
