import torch
import numpy as np
import json
import glob
import os

# Load parameter data directly
datadir = 'datasets/polymax_dataset/raw'
data_files = glob.glob(os.path.join(datadir, '*.npz'))

params_check = []
for file_path in data_files[:10]:  # Check first 10 files
    if file_path.endswith('.npz'):
        data = np.load(file_path)
        if 'params' in data:
            params_tensor = torch.from_numpy(data['params']).float()
            if params_tensor.dim() == 1:
                params_tensor = params_tensor.unsqueeze(0)
            params_check.append(params_tensor)
        print(f"File {os.path.basename(file_path)} keys: {list(data.keys())}")
    elif file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            json_data = json.load(f)
        if 'parameter_vector' in json_data:
            param_vector = json_data['parameter_vector']
            params_check.append(torch.Tensor(param_vector).unsqueeze(0))

if params_check:
    full_params = torch.cat(params_check, dim=0)
    params_std = torch.std(full_params, dim=0)
    params_mean = torch.mean(full_params, dim=0)
    
    print(f"Parameter shape: {full_params.shape}")
    print(f"Params mean (first 10): {params_mean[:10]}")
    print(f"Params std (first 10): {params_std[:10]}")
    print(f"Min std: {torch.min(params_std)}")
    print(f"Max std: {torch.max(params_std)}")
    print(f"Any zero std: {torch.any(params_std == 0)}")
    print(f"Any NaN in mean: {torch.any(torch.isnan(params_mean))}")
    print(f"Any NaN in std: {torch.any(torch.isnan(params_std))}")
    print(f"Any inf in mean: {torch.any(torch.isinf(params_mean))}")
    print(f"Any inf in std: {torch.any(torch.isinf(params_std))}")
    
    # Check raw parameter ranges
    print(f"Raw param min: {torch.min(full_params)}")
    print(f"Raw param max: {torch.max(full_params)}")
else:
    print("No parameter data found")