"""Dataset utilities for PolyMAX parameter prediction training."""
import json
import csv
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

class PolyMAXDataset(Dataset):
    """Load feature vectors and corresponding parameter vectors.

    Args:
        manifest_csv: CSV with columns split,audio_path,params_path, ...
        features_dir: Directory containing .npy feature files.
        split: One of {"train","val","test"}.
        schema_json: JSON schema describing parameter field order.
    Returns:
        feature (torch.FloatTensor), params (torch.FloatTensor)
    """
    def __init__(self, manifest_csv: str, features_dir: str, split: str,
                 schema_json: str):
        self.features_dir = Path(features_dir)
        self.rows = []
        with open(manifest_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] == split:
                    self.rows.append(row)
        schema = json.load(open(schema_json))
        self.order = [f["name"] for f in schema["fields"]]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[idx]
        feat = np.load(self.features_dir / (Path(row["audio_path"]).stem + ".npy")).astype(np.float32)
        params = json.load(open(row["params_path"]))
        if isinstance(params, dict):
            params = np.array([params[n] for n in self.order], dtype=np.float32)
        else:
            params = np.array(params, dtype=np.float32)
        return torch.from_numpy(feat), torch.from_numpy(params)
