#!/usr/bin/env python3
import json, argparse, numpy as np, torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.nn.functional import l1_loss, mse_loss
from pathlib import Path
from surrogate_mlp import SurrogateMLP

class ParamsToFeatDS(Dataset):
    def __init__(self, manifest_csv, features_dir, schema_json, split="train"):
        import csv
        self.rows = [r for r in csv.DictReader(open(manifest_csv)) if r["split"]==split]
        self.features_dir = Path(features_dir)
        self.schema = json.load(open(schema_json))
        self.order  = [f["name"] for f in self.schema["fields"]]
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        import json, numpy as np
        r = self.rows[i]
        feat = np.load(self.features_dir / (Path(r["audio_path"]).stem + ".npy")).astype("float32")
        params = json.load(open(r["params_path"]))
        if isinstance(params, dict):
            params = np.array([params[n] for n in self.order], dtype=np.float32)
        else:
            params = np.array(params, dtype=np.float32)
        return torch.from_numpy(params), torch.from_numpy(feat)

def main(manifest, features_dir, schema_json, out_path):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tr = ParamsToFeatDS(manifest, features_dir, schema_json, "train")
    va = ParamsToFeatDS(manifest, features_dir, schema_json, "val")
    tl = DataLoader(tr, batch_size=256, shuffle=True)
    vl = DataLoader(va, batch_size=256, shuffle=False)
    out_dim = tr[0][1].numel()

    model = SurrogateMLP(in_dim=66, out_dim=out_dim).to(device)
    opt = AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    best = 1e9; Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    for ep in range(200):
        model.train(); tr_loss = 0.0
        for p,f in tl:
            p,f = p.to(device), f.to(device)
            opt.zero_grad()
            pred = model(p)
            loss = 0.7*l1_loss(pred,f) + 0.3*mse_loss(pred,f)
            loss.backward(); opt.step()
            tr_loss += loss.item()*p.size(0)
        tr_loss /= len(tr)

        model.eval(); va_loss = 0.0
        with torch.no_grad():
            for p,f in vl:
                p,f = p.to(device), f.to(device)
                pred = model(p)
                va_loss += (0.7*l1_loss(pred,f) + 0.3*mse_loss(pred,f)).item()*p.size(0)
        va_loss /= len(va)
        print(f"[surrogate] ep{ep:03d} tr {tr_loss:.4f} va {va_loss:.4f}")
        if va_loss < best - 1e-4:
            best = va_loss
            torch.save({"model": model.state_dict(), "out_dim": out_dim}, out_path)
        if ep>5 and va_loss>best+0.02: break

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--features_dir", default="data/features")
    ap.add_argument("--schema_json", default="data/params_schema.json")
    ap.add_argument("--out_path", default="checkpoints/surrogate.pt")
    args = ap.parse_args()
    main(**vars(args))
