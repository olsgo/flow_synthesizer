import argparse
import torch
import json, ast
#from matplotlib import pyplot as plt
from osc_server import FlowServer
from osc_utils import generate_dataset
from utils.data import load_dataset
from evaluate import evaluate_dimensions, evaluate_dataset
from torch.utils.data import DataLoader
import numpy as np
import os
from os.path import expanduser

# Debug mode
__DEBUG__ = False

parser = argparse.ArgumentParser()
parser.add_argument('--in_port',    type=int,   default=1232)
parser.add_argument('--out_port',   type=int,   default=1233)
parser.add_argument('--ip',         type=str,   default="127.0.0.1")
# Model arguments
parser.add_argument('--model_path', type=str,   default="results/")
parser.add_argument('--model_name', type=str,   default="vae_mel_mse_32_cnn_mlp_1.model")
# Data arguments
parser.add_argument('--path',       type=str,   default='data',     help='')
parser.add_argument('--dataset',    type=str,   default='64par',    help='')
parser.add_argument('--train_type', type=str,   default='fixed',    help='')
parser.add_argument('--data',       type=str,   default='mel',      help='')
parser.add_argument('--projection', type=str,   default='pca',      help='')
parser.add_argument('--batch_size', type=int,   default=128,        help='')
parser.add_argument('--nbworkers',  type=int,   default=0,          help='')
parser.add_argument('--reanalyze',  type=int,   default=0,          help='')
parser.add_argument('--device',     type=str,   default='cpu',      help='')
parser.add_argument('--synth_type',  type=str,   default=None,       help='Synthesizer type (diva, massive_x)')
parser.add_argument('--plugin_path', type=str,   default=None,       help='Path to VST/AU plugin')
parser.add_argument('--list_synths', action='store_true',            help='List available synthesizer types and exit')
args = parser.parse_args()

# Handle listing synthesizers
if args.list_synths:
    from synth.synthesizer_interface import SynthesizerFactory
    print("Available synthesizer types:")
    for synth_type in SynthesizerFactory.get_supported_synthesizers():
        print(f"  - {synth_type}")
    exit(0)

#%%
"""
###################
Load model
###################
"""
model = None
args.model = args.model_path + args.dataset + '/' + args.model_name
if args.model:
    model = torch.load(args.model, map_location="cpu", weights_only=False)
    model = model.eval()

"""
###################
Load dataset
###################
""" 
print('[Loading dataset for ' + args.data + ']')
if (args.train_type == 'random'):
    train_loader, valid_loader, test_loader, args = load_dataset(args)
else:
    ref_split = args.path + '/reference_split_' + args.dataset + "_" + args.data + '.th'
    print('[About to load]')
    data = torch.load(ref_split, weights_only=False)
    train_loader, valid_loader, test_loader = data[0], data[1], data[2]
    print('[Changing refs in reference]')
    home = expanduser("~")
    for t in [train_loader, valid_loader, test_loader]:
        t.dataset.datadir = home + '/Datasets/diva_dataset/' + args.dataset
        t.dataset.trans_datasets[args.data].datadir = home + '/Datasets/diva_dataset/' + args.dataset
    torch.save([train_loader, valid_loader, test_loader], ref_split)
# Remove the shuffling from dataset
train_loader = DataLoader(train_loader.dataset, batch_size=64, shuffle=False, num_workers=2)
valid_loader = DataLoader(valid_loader.dataset, batch_size=64, shuffle=False, num_workers=2)
test_loader = DataLoader(test_loader.dataset, batch_size=64, shuffle=False, num_workers=2)

    
#%% Combine sets    
audioset = [train_loader, valid_loader, test_loader]

# Import configuration and synthesizer interface
from config_manager import config
from synth.synthesizer_interface import SynthesizerFactory

# Determine synthesizer type
synth_type = args.synth_type or config.get_synthesizer_type()
print(f'[Using synthesizer: {synth_type}]')

# Create synthesizer interface to get parameters
plugin_path = args.plugin_path or config.get_plugin_path(synth_type)
synth_interface = SynthesizerFactory.create_synthesizer(synth_type, plugin_path)

# Load parameter mapping and defaults
param_mapping = synth_interface.load_parameter_mapping()
rev_idx = synth_interface.create_reverse_mapping(param_mapping)
params_default = synth_interface.load_default_parameters(args.dataset)

param_dict = params_default
param_names = test_loader.dataset.final_params
print('[Reference set on which model was trained]')
print(param_names)
print(f'[Parameter mapping loaded for {synth_type}: {len(param_mapping)} parameters]')

#%%
"""
###################
Perform model pre-analysis
###################
""" 
# Target file of analysis
analysis_file = args.model.replace('.model', '.analysis')
# Target dataset
dataset_file = args.model.replace('.model', '.dataset')
if (len(args.projection) > 0):
    analysis_file += '.' + args.projection
    dataset_file += '.' + args.projection
# Create analysis files
if (not os.path.exists(analysis_file + '.npy') or args.reanalyze):
    # Perform dataset evaluation
    final_z, final_meta, pca, z_vars, z_means = evaluate_dataset(model, [train_loader, valid_loader, test_loader], args)
    # Perform dimension evaluation    
    d_idx, d_vars, d_params, d_desc, desc_max =  evaluate_dimensions(model, pca, args)
    # Save information
    model_analysis = {
     'd_idx':d_idx, 
     'd_vars':d_vars, 
     'd_params':d_params, 
     'd_desc':d_desc, 
     'desc_max':desc_max,
     'final_z':final_z,
     'final_meta':final_meta, 
     'pca':pca, 
     'z_vars':z_vars, 
     'z_means':z_means
     }
    # Generate offline presets dataset
    model_analysis = generate_dataset(dataset_file + '.txt', [train_loader, valid_loader, test_loader], model_analysis)
    # Keep path to the model dataset
    model_analysis['dataset_path'] = dataset_file + '.txt'
    # Save the whole analysis
    np.save(analysis_file, model_analysis)
else:
    model_analysis = np.load(analysis_file + '.npy', allow_pickle=True).item()

#%%
"""
###################
Create server
###################
"""
server = FlowServer(args.in_port, args.out_port, model=model, dataset=audioset, data=args.data, param_names=param_names, param_dict=param_dict, analysis=model_analysis, debug=__DEBUG__, args=args, synth_type=synth_type)
#%%
if (__DEBUG__):
    # Test pitch analysis
    print('[Debug mode : Testing server on given functions]')
else:
    print('[Running server on ports in : %d - out : %d]'%(args.in_port, args.out_port))
    server.run()