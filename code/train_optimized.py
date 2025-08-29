#!/usr/bin/env python3
"""
Optimized training script for Flow Synthesizer with focus on maximum resynthesis accuracy.
Key optimizations:
1. L1 loss instead of MSE for better parameter regression
2. Higher learning rate with cosine annealing
3. Increased model capacity
4. Early regression start for better parameter learning
5. Enhanced data augmentation
"""

import matplotlib
matplotlib.use('agg')
import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.data import load_dataset, get_external_sounds
from models.vae.ae import AE, RegressionAE, DisentanglingAE
from models.vae.vae import VAE
from models.vae.wae import WAE
from models.vae.vae_flow import VAEFlow
from models.loss import multinomial_loss, multinomial_mse_loss
from models.basic import GatedMLP, GatedCNN, construct_encoder_decoder, construct_flow, construct_regressor, construct_disentangle
from evaluate import evaluate_model

# Optimized argument parser
parser = argparse.ArgumentParser()
# Dataset arguments
parser.add_argument('--path',           type=str,   default='',             help='Path to the dataset')
parser.add_argument('--test_sounds',    type=str,   default='',             help='Path to test sounds')
parser.add_argument('--output',         type=str,   default='outputs_optimized',      help='Path to output directory')
parser.add_argument('--dataset',        type=str,   default='polymax_dataset',        help='Name of the dataset')
parser.add_argument('--data',           type=str,   default='mel',          help='Type of data to train on')
parser.add_argument('--train_type',     type=str,   default='fixed',        help='Fixed or random data split')
parser.add_argument('--nbworkers',      type=int,   default=0,              help='Number of workers for parallel import')

# Model arguments - OPTIMIZED FOR ACCURACY
parser.add_argument('--model',          type=str,   default='vae',          help='Type of model (MLP, CNN, AE, VAE, WAE)')
parser.add_argument('--loss',           type=str,   default='l1',           help='Loss for parameter regression - L1 for better accuracy')
parser.add_argument('--rec_loss',       type=str,   default='l1',           help='Reconstruction loss - L1 for better accuracy')
parser.add_argument('--n_classes',      type=int,   default=61,             help='Classes for multinoulli loss')
parser.add_argument('--n_hidden',       type=int,   default=2048,           help='Number of hidden units - INCREASED')
parser.add_argument('--n_layers',       type=int,   default=6,              help='Number of computing layers - INCREASED')

# CNN parameters - ENHANCED
parser.add_argument('--channels',       type=int,   default=128,            help='Number of channels in convolution - INCREASED')
parser.add_argument('--kernel',         type=int,   default=5,              help='Size of convolution kernel')
parser.add_argument('--dilation',       type=int,   default=3,              help='Dilation factor of convolution')

# AE-specific parameters - OPTIMIZED
parser.add_argument('--layers',         type=str,   default='gated_cnn',    help='Type of layers in the model')
parser.add_argument('--encoder_dims',   type=int,   default=128,            help='Number of encoder output dimensions - INCREASED')
parser.add_argument('--latent_dims',    type=int,   default=0,              help='Number of latent dimensions')
parser.add_argument('--warm_latent',    type=int,   default=25,             help='Warmup epochs for latent - REDUCED')
parser.add_argument('--start_regress',  type=int,   default=10,             help='Epoch to start regression - EARLY START')
parser.add_argument('--warm_regress',   type=int,   default=50,             help='Warmup epochs for regression - REDUCED')
parser.add_argument('--beta_factor',    type=int,   default=1,              help='Beta factor in VAE')

# Two-step training parameters
parser.add_argument('--ref_model',      type=str,   default='',             help='Reference model')

# Flow specific parameters
parser.add_argument('--flow',           type=str,   default='iaf',          help='Type of flow to use')
parser.add_argument('--flow_length',    type=int,   default=16,             help='Number of flow transforms')

# Regression parameters - ENHANCED
parser.add_argument('--regressor',      type=str,   default='mlp',          help='Type of regressor')
parser.add_argument('--reg_layers',     type=int,   default=4,              help='Number of regression layers - INCREASED')
parser.add_argument('--reg_hiddens',    type=int,   default=512,            help='Number of units in regressor - INCREASED')
parser.add_argument('--reg_flow',       type=str,   default='maf',          help='Type of flow in regressor')
parser.add_argument('--reg_factor',     type=float, default=5e3,            help='Regression loss weight - INCREASED')

# Optimization arguments - OPTIMIZED
parser.add_argument('--k_run',          type=int,   default=0,              help='ID of runs (k-folds)')
parser.add_argument('--early_stop',     type=int,   default=40,             help='Early stopping - REDUCED')
parser.add_argument('--plot_interval',  type=int,   default=50,             help='Interval of plotting frequency')
parser.add_argument('--batch_size',     type=int,   default=16,             help='Size of the batch - REDUCED for stability')
parser.add_argument('--epochs',         type=int,   default=150,            help='Number of epochs to train on')
parser.add_argument('--eval',           type=int,   default=50,             help='Frequency of full evalution')
parser.add_argument('--lr',             type=float, default=5e-4,           help='Learning rate - OPTIMIZED')
parser.add_argument('--max_gamma',      type=float, default=500.0,          help='Maximum gamma cap for regression loss weight')

# Semantic arguments
parser.add_argument('--semantic_dim',   type=int,   default=-1,             help='Using semantic dimension')
parser.add_argument('--dis_layers',     type=int,   default=8,              help='Number of disentangling layers')
parser.add_argument('--disentangling',  type=str,   default='density',      help='Type of disentangling approach')
parser.add_argument('--start_disentangle',type=int, default=100,            help='Epoch to start disentangling')
parser.add_argument('--warm_disentangle',type=int,  default=25,             help='Warmup on disentanglement')

# Evaluation parameters
parser.add_argument('--batch_evals',    type=int,   default=16,             help='Number of batch to evaluate')
parser.add_argument('--batch_out',      type=int,   default=3,              help='Number of batch to synthesize')
parser.add_argument('--check_exists',   type=int,   default=0,              help='Check if model exists')
parser.add_argument('--time_limit',     type=int,   default=0,              help='Maximum time to train (in minutes)')

# CUDA arguments
parser.add_argument('--device',         type=str,   default='cpu',          help='Device for CUDA')
parser.add_argument('--synth_type',     type=str,   default='polymax',      help='Synthesizer type')

args = parser.parse_args()
start_time = time.time()

args.synthesize = False

# Set default paths
if (len(args.path) == 0):
    args.path = 'datasets'
    args.test_sounds = ''
    
if (args.device not in ['cpu']):
    args.synthesize = True
if (args.device != 'cpu'):
    torch.backends.cudnn.benchmark=True

print('Arguments:')
for arg in vars(args):
    print('  ' + arg + ': ' + str(getattr(args, arg)))

# Create output directories
if not os.path.exists('{0}'.format(args.output)):
    os.makedirs('{0}'.format(args.output))
    os.makedirs('{0}/audio'.format(args.output))
    os.makedirs('{0}/images'.format(args.output))
    os.makedirs('{0}/models'.format(args.output))

# Model naming
model_name = '{0}_{1}_{2}_{3}_optimized'.format(args.model, args.data, args.loss, str(args.latent_dims))
if (not (args.model in ['mlp', 'gated_mlp', 'cnn', 'gated_cnn', 'res_cnn'])):
    model_name += '_' + args.layers
    if (args.model == 'vae_flow'):
        model_name += '_' + args.flow
    model_name += '_' + args.regressor
    if (args.regressor != 'mlp'):
        model_name += '_' + args.reg_flow + '_' + str(args.reg_layers)
    if (args.semantic_dim > -1):
        model_name += '_' + str(args.semantic_dim) + '_' + args.disentangling
if (args.k_run > 0):
    model_name += '_' + str(args.k_run)
    
base_dir = '{0}/'.format(args.output)
base_img = '{0}/images/{1}'.format(args.output, model_name)
base_audio = '{0}/audio/{1}'.format(args.output, model_name)

if (args.check_exists == 1):
    if os.path.exists(args.output + '/models/' + model_name + '.synth.results.npy'):
        print('[Found ' + args.output + '/models/' + model_name + '.synth.results.npy - Exiting.]')
        exit

device_str = str(args.device).lower()
if device_str == 'cuda' and torch.cuda.is_available():
    args.device = torch.device('cuda')
elif device_str == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    args.device = torch.device('mps')
else:
    args.device = torch.device('cpu')
print('Optimization will be on ' + str(args.device) + '.')

print('[Loading dataset]')
ref_split = args.path + '/reference_split_' + args.dataset+ "_" + args.data + '.th'
if (args.train_type == 'random' or (not os.path.exists(ref_split))):
    train_loader, valid_loader, test_loader, args = load_dataset(args)
    if (args.train_type == 'fixed'):
        torch.save([train_loader, valid_loader, test_loader], ref_split)
    fixed_data, fixed_params, fixed_meta, fixed_audio = next(iter(test_loader))
    fixed_data, fixed_params, fixed_meta, fixed_audio = fixed_data.to(args.device), fixed_params.to(args.device), fixed_meta, fixed_audio
    fixed_batch = (fixed_data, fixed_params, fixed_meta, fixed_audio)
else:
    data = torch.load(ref_split)
    train_loader, valid_loader, test_loader = data[0], data[1], data[2]
    fixed_data, fixed_params, fixed_meta, fixed_audio = next(iter(test_loader))
    fixed_data, fixed_params, fixed_meta, fixed_audio = fixed_data.to(args.device), fixed_params.to(args.device), fixed_meta, fixed_audio
    fixed_batch = (fixed_data, fixed_params, fixed_meta, fixed_audio)
    args.output_size = train_loader.dataset.output_size
    args.input_size = train_loader.dataset.input_size

if (args.latent_dims == 0):
    args.latent_dims = args.output_size

print('Dataset loaded:')
print('  Input size: ' + str(args.input_size))
print('  Output size: ' + str(args.output_size))
print('  Latent dims: ' + str(args.latent_dims))

print('[Creating optimized model for maximum resynthesis accuracy]')
if (args.loss in ['multinomial']):
    args.output_size *= args.n_classes
if (args.loss in ['multi_mse']):
    args.output_size *= (args.n_classes + 1)
    
if (args.model == 'mlp'):
    model = GatedMLP(np.prod(args.input_size), args.output_size, hidden_size = args.n_hidden, n_layers = args.n_layers, type_mod='normal')
elif (args.model == 'gated_mlp'): 
    model = GatedMLP(np.prod(args.input_size), args.output_size, hidden_size = args.n_hidden, n_layers = args.n_layers, type_mod='gated')
elif (args.model == 'cnn'):
    model = GatedCNN(args.input_size, args.output_size, channels = args.channels, n_layers = 4, hidden_size = args.n_hidden, n_mlp = 3, type_mod='normal', args=args)
elif (args.model == 'gated_cnn'):
    model = GatedCNN(args.input_size, args.output_size, channels = args.channels, n_layers = 4, hidden_size = args.n_hidden, n_mlp = 3, type_mod='gated', args=args)
elif (args.model == 'res_cnn'):
    model = GatedCNN(args.input_size, args.output_size, channels = args.channels, n_layers = 4, hidden_size = args.n_hidden, n_mlp = 3, type_mod='residual', args=args)
elif (args.model in ['ae', 'vae', 'wae', 'vae_flow']):
    # Reconstruction loss
    if (args.rec_loss == 'mse'):
        rec_loss = nn.MSELoss(reduction='sum').to(args.device)
    elif (args.rec_loss == 'l1'):
        rec_loss = nn.SmoothL1Loss(reduction='sum').to(args.device)
    elif (args.rec_loss == 'multinomial'):
        rec_loss = multinomial_loss
    elif (args.rec_loss == 'multi_mse'):
        rec_loss = multinomial_mse_loss
    else:
        raise Exception('Unknown reconstruction loss ' + args.rec_loss)
    
    # Enhanced encoder/decoder with more capacity
    encoder, decoder = construct_encoder_decoder(args.input_size, args.encoder_dims, args.latent_dims, channels = args.channels, n_layers = args.n_layers, hidden_size = args.n_hidden, n_mlp = args.n_layers // 2, type_mod=args.layers, args=args)
    
    if (args.model == 'ae'):
        model = AE(encoder, decoder, args.encoder_dims, args.latent_dims)
    elif (args.model == 'vae'):
        model = VAE(encoder, decoder, args.input_size, args.encoder_dims, args.latent_dims)
    elif (args.model == 'wae'):
        model = WAE(encoder, decoder, args.input_size, args.encoder_dims, args.latent_dims)
    elif (args.model == 'vae_flow'):
        flow, blocks = construct_flow(args.latent_dims, flow_type=args.flow, flow_length=args.flow_length, amortization='input')
        model = VAEFlow(encoder, decoder, flow, args.input_size, args.encoder_dims, args.latent_dims)
    
    # Enhanced regressor with more capacity
    regression_model = construct_regressor(args.latent_dims, args.output_size, model=args.regressor, hidden_dims = args.reg_hiddens, n_layers=args.reg_layers, flow_type=args.reg_flow)
    if (args.semantic_dim == -1):
        model = RegressionAE(model, args.latent_dims, args.output_size, rec_loss, regressor=regression_model, regressor_name=args.regressor)
    else:
        disentangling = construct_disentangle(args.latent_dims, model=args.disentangling, semantic_dim=args.semantic_dim, n_layers=args.dis_layers, flow_type=args.reg_flow)
        model = DisentanglingAE(model, args.latent_dims, args.output_size, rec_loss, regressor=regression_model, regressor_name=args.regressor, disentangling=disentangling, semantic_dim=args.semantic_dim)
else:
    raise Exception('Unknown model ' + args.model)

model = model.to(args.device)
print('Model created with enhanced capacity for maximum accuracy')

# Load reference model if specified
if (len(args.ref_model) > 0):
    print('[Loading reference ' + args.ref_model + ']')
    ref_model = torch.load(args.ref_model)
    if (args.regressor != 'mlp'):
        ref_model_ae = ref_model.ae_model.to(args.device)
        model.ae_model = None
        model.ae_model = ref_model_ae
        ref_model = None
    else:
        model = None
        model = ref_model.to(args.device)

print('[Setting up optimized training]')
# Enhanced optimizer with weight decay for regularization
optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

# Cosine annealing scheduler for better convergence
scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

# Loss function - L1 for better parameter accuracy
if (args.loss == 'mse'):
    loss = nn.MSELoss(reduction='mean').to(args.device)
elif (args.loss == 'l1'):
    loss = nn.SmoothL1Loss(reduction='mean').to(args.device)
elif (args.loss == 'bce'):
    loss = nn.BCELoss(reduction='mean').to(args.device)
elif (args.loss == 'multinomial'):
    loss = multinomial_loss
elif (args.loss == 'multi_mse'):
    loss = multinomial_mse_loss
else:
    raise Exception('Unknown loss ' + args.loss)

print('Training setup:')
print('  Loss function: ' + args.loss + ' (optimized for accuracy)')
print('  Optimizer: AdamW with weight decay')
print('  Scheduler: Cosine annealing')
print('  Early regression start: epoch ' + str(args.start_regress))
print('  Enhanced model capacity: ' + str(args.n_hidden) + ' hidden units, ' + str(args.channels) + ' channels')

# Training loop with enhanced monitoring
losses = torch.zeros(args.epochs, 3)
if (args.epochs == 0):
    losses = torch.zeros(200, 3)
best_loss = np.inf
early = 0

print('[Starting optimized training for maximum resynthesis accuracy]')
for i in range(args.epochs):
    # Enhanced beta and gamma scheduling for better parameter learning
    args.beta = args.beta_factor * (float(i) / float(max(args.warm_latent, i)))
    if (i >= args.start_regress):
        args.gamma = ((float(i - args.start_regress) * args.reg_factor) / float(max(args.warm_regress, i - args.start_regress)))
        if (args.regressor != 'mlp'):
            args.gamma *= 1e-1
    else:
        args.gamma = 0
    # Cap gamma for stability
    if args.max_gamma > 0:
        args.gamma = min(args.gamma, args.max_gamma)
    if (i >= args.start_disentangle):
        args.delta = ((float(i - args.start_disentangle)) / float(max(args.warm_disentangle, i - args.start_disentangle)))
    else:
        args.delta = 0
    
    print('Epoch %d - Beta: %.3f, Gamma: %.3f, LR: %.6f'%(i, args.beta, args.gamma, optimizer.param_groups[0]['lr']))
    
    # Training
    losses[i, 0] = model.train_epoch(train_loader, loss, optimizer, args)    
    
    # Validation
    losses[i, 1] = model.eval_epoch(valid_loader, loss, args)
    
    # Update learning rate with cosine annealing
    if ((not args.model in ['ae', 'vae', 'wae', 'vae_flow']) or (i >= args.start_regress)):
        scheduler.step()
    
    # Test evaluation
    losses[i, 2] = model.eval_epoch(test_loader, loss, args)
    if (args.start_regress == 1000):
        losses[i, 1] = losses[i, 0]
        losses[i, 2] = losses[i, 0]
    
    # Enhanced model saving with better validation tracking
    if (losses[i, 1] < best_loss):
        best_loss = losses[i, 1]
        model_dir = args.output + '/models/'
        os.makedirs(model_dir, exist_ok=True)
        torch.save(model, model_dir + model_name + '.model')
        # Also save/update a canonical checkpoint used by integration tests/tools
        torch.save(model, model_dir + 'best_model.pth')
        print('  -> New best model saved (validation loss: %.6f)' % best_loss)
        early = 0
    elif (args.early_stop > 0 and i >= args.start_regress):
        early += 1
        if (early > args.early_stop):
            print('[Model stopped early after %d epochs without improvement]' % args.early_stop)
            break
    
    # Enhanced evaluation and plotting
    if ((i + 1) % args.plot_interval == 0 or (args.epochs == 1)):
        args.plot = 'train'
        with torch.no_grad():
            model.eval()
            evaluate_model(model, fixed_batch, test_loader, args, train=True, name=base_img + '_batch_' + str(i))
    
    # Time limit check
    if ((args.time_limit > 0) and (((time.time() - start_time) / 60.0) > args.time_limit)):
        print('[Hitting time limit after ' + str((time.time() - start_time) / 60.0) + ' minutes.]')
        print('[Going to evaluation mode]')
        break
    
    print('  Train Loss: %.6f, Valid Loss: %.6f, Test Loss: %.6f' % (losses[i, 0], losses[i, 1], losses[i, 2]))
    torch.cuda.empty_cache()

print('[Training completed - loading best model for evaluation]')

# Final evaluation with enhanced metrics
from evaluate import evaluate_params, evaluate_synthesis, evaluate_projection
from evaluate import evaluate_reconstruction, evaluate_latent_space
from evaluate import evaluate_meta_parameters, evaluate_semantic_parameters
from evaluate import evaluate_latent_neighborhood

args.plot = 'final'
args.model_name, args.base_img, args.base_audio = model_name, base_img, base_audio
args.base_model = args.output + '/models/' + model_name

print('[Reload best performing model]')
model = torch.load(args.output + '/models/' + model_name + '.model')
model = model.to(args.device)

print('[Performing comprehensive evaluation for resynthesis accuracy]')
with torch.no_grad():
    evaluate_params(model, test_loader, args, losses=losses)
    
    if (args.synthesize):
        from synth.synthesize import create_synth
        print('[Synthesis evaluation]')
        args.engine, args.generator, args.param_defaults, args.rev_idx = create_synth(args.dataset, args.synth_type)
    
    if (args.model not in ['mlp', 'gated_mlp', 'cnn', 'gated_cnn', 'res_cnn']):
        evaluate_reconstruction(model, test_loader, args, train=False)
        args = evaluate_latent_space(model, test_loader, args, train=False)
        evaluate_meta_parameters(model, test_loader, args, train=False)
        evaluate_latent_neighborhood(model, test_loader, args, train=False)
        evaluate_semantic_parameters(model, test_loader, args, train=False)
    
    if (args.synthesize):
        evaluate_synthesis(model, test_loader, args, train=False)
        if args.test_sounds:
            print('[Load set of testing sound (outside dataset)]')
            test_sounds = get_external_sounds(args.test_sounds, test_loader.dataset, args)
            evaluate_projection(model, test_sounds, args, train=False)

print('[Optimized training completed successfully]')
print('Best validation loss: %.6f' % best_loss)
print('Model saved as: ' + args.output + '/models/' + model_name + '.model')
print('Training time: %.2f minutes' % ((time.time() - start_time) / 60.0))
