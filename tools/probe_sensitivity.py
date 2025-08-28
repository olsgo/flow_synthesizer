#!/usr/bin/env python3
# Estimate each parameter's perceptual impact by random sampling small deltas
# and measuring feature distance. Save as weights for training.
import json, argparse, numpy as np

def main(schema_json, out_json, distances_csv):
    import csv
    schema = json.load(open(schema_json))
    names = [f["name"] for f in schema["fields"]]
    # Load precomputed distances for small +/- deltas per param (you can generate with Pedalboard)
    # Format: CSV rows: param_name, mean_delta_feat_dist
    d = {r[0]: float(r[1]) for r in csv.reader(open(distances_csv))}
    # Normalize weights to mean=1.0
    vals = np.array([d.get(n, 1.0) for n in names], dtype=np.float32)
    vals = vals / (vals.mean() + 1e-8)
    json.dump(vals.tolist(), open(out_json,"w"), indent=2)
    print("Saved sensitivity weights to", out_json)

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema_json", default="data/params_schema.json")
    ap.add_argument("--out_json", default="data/sensitivity_weights.json")
    ap.add_argument("--distances_csv", required=True)
    args = ap.parse_args()
    main(**vars(args))
