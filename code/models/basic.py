# -*- coding: utf-8 -*-

import torch
import numpy as np
import torch.nn as nn
from torch.distributions import MultivariateNormal
# Model layers imports
from models.layers import GatedDense, GatedConv2d, ResConv2d
from models.layers import GatedConvTranspose2d, ResConvTranspose2d 
# Flows library imports
from models.flows.coupling import MaskedCouplingFlow
from models.flows.flow import NormalizingFlow
from models.flows.iaf import IAFlow, ContextIAFlow, DDSF_IAFlow
from models.flows.maf import MAFlow, ContextMAFlow
from models.flows.naf import DeepSigmoidFlow, DeepDenseSigmoidFlow
from models.flows.normalization import BatchNormFlow
from models.flows.order import ReverseFlow, ShuffleFlow
from models.flows.planar import PlanarFlow
from models.flows.sylvester import TriangularSylvesterFlow
# Regressors import
from models.regression import BayesianRegressor
from models.regression import FlowTransform, FlowKL, FlowKLFull, FlowCDE
from models.regression import FlowExternal, FlowPosterior, FlowDecoder
# Disentangling import
from models.disentangling import DisentanglingFlow

def construct_encoder_decoder(in_size, enc_size, latent_size, hidden_size = 512, channels = 32, n_layers = 6, n_mlp = 2, type_ae = 'ae', type_mod='gated_cnn', args=None):
    """ Construct encoder and decoder layers for AE models """
    # MLP layers
    if (type_mod in ['mlp', 'gated_mlp']):
        type_ed = (type_mod == 'mlp') and 'normal' or 'gated'
        encoder = GatedMLP(np.prod(in_size), enc_size, hidden_size, n_layers, type_ed)
        decoder = DecodeMLP(latent_size, in_size, hidden_size, n_layers, type_ed)
    elif (type_mod in ['cnn', 'gated_cnn', 'res_cnn']):
        type_ed = (type_mod == 'cnn') and 'normal' or ((type_mod == 'res_cnn') and 'residual' or 'gated')
        encoder = GatedCNN(in_size, enc_size, channels, n_layers, hidden_size, n_mlp, type_ed, args)
        decoder = DecodeCNN(latent_size, encoder.cnn_size, in_size, channels, n_layers, hidden_size, n_mlp, type_ed, args)
    else:
        # Default to gated CNN if type_mod is not recognized
        type_ed = 'gated'
        encoder = GatedCNN(in_size, enc_size, channels, n_layers, hidden_size, n_mlp, type_ed, args)
        decoder = DecodeCNN(latent_size, encoder.cnn_size, in_size, channels, n_layers, hidden_size, n_mlp, type_ed, args)
    return encoder, decoder

def construct_flow(flow_dim, flow_type='maf', flow_length=16, amortization='input'):
    """ Construct normalizing flow """
    if flow_type == 'planar':
        blocks = [ PlanarFlow ]
    elif flow_type == 'sylvester':
        blocks = [ TriangularSylvesterFlow, BatchNormFlow, ShuffleFlow ]
    elif flow_type == 'real_nvp':
        blocks = [ MaskedCouplingFlow, BatchNormFlow, ShuffleFlow ]
    elif flow_type == 'maf':
        blocks = [ MAFlow, BatchNormFlow, ReverseFlow ]
    elif flow_type == 'iaf':
        blocks = [ IAFlow, BatchNormFlow, ShuffleFlow ]
    elif flow_type == 'dsf':
        blocks = [ DeepSigmoidFlow, BatchNormFlow, ReverseFlow ]
    elif flow_type == 'ddsf':
        blocks = [ DeepDenseSigmoidFlow, BatchNormFlow, ReverseFlow ]
    elif flow_type == 'ddsf_iaf':
        blocks = [ DDSF_IAFlow, BatchNormFlow, ShuffleFlow ]
    elif flow_type == 'iaf_ctx':
        blocks = [ ContextIAFlow, BatchNormFlow, ShuffleFlow ]
    elif flow_type == 'maf_ctx':
        blocks = [ ContextMAFlow, BatchNormFlow, ReverseFlow ]
    else:
        raise ValueError('Invalid flow choice : ' + flow_type)
    flow = NormalizingFlow(
        dim=flow_dim, blocks=blocks, flow_length=flow_length,
        density=MultivariateNormal(torch.zeros(flow_dim),
        torch.eye(flow_dim)), amortized='self'
    )
    return flow, blocks

def construct_disentangle(in_dims, model='density', semantic_dim=0, n_layers=4, flow_type='maf'):
    # Construct the flow
    _, blocks = construct_flow(in_dims, flow_type=flow_type, flow_length=1, amortization='self')
    # Construct disentangling model
    if (model in ['density', 'base']):
        dis_model = DisentanglingFlow(in_dims, blocks=blocks, flow_length=n_layers, amortize='self', eps_var=1, var_type='dims_rand')
    elif (model == 'full'):
        dis_model = DisentanglingFlow(in_dims, blocks=blocks, flow_length=n_layers, amortize='self', eps_var=1e-1, var_type='dims_rand')
    return dis_model

def construct_regressor(in_dims, out_dims, model='mlp', hidden_dims = 0, n_layers = 16, flow_type='maf', amortize='self', eps_var=1e-2, var_type = 'dims'):
    if (hidden_dims == 0):
        hidden_dims = in_dims * 4
    # MLP Regressor
    if (model == 'mlp'):
        regression_model = nn.Sequential()
        for l in range(n_layers):
            in_s = (l == 0) and in_dims or hidden_dims
            out_s = (l == (n_layers - 1)) and out_dims or hidden_dims
            regression_model.add_module('l%d'%l, nn.Linear(in_s, out_s))
            if (l < (n_layers - 1)):
                regression_model.add_module('b%d'%l, nn.BatchNorm1d(out_s))
                regression_model.add_module('r%d'%l, nn.ReLU())
                regression_model.add_module('d%d'%l, nn.Dropout(p=.3))
    # Bayesian regression
    elif (model == 'bnn'):
        _, blocks = construct_flow(in_dims, flow_type=flow_type, flow_length=1, amortization=amortize)
        regression_model = BayesianRegressor(in_dims, out_dims, hidden_size=hidden_dims, n_layers=n_layers, blocks=blocks, flow_length=n_layers)
    # Flow mixture prediction
    elif (model[:4] == 'flow'):
        _, blocks = construct_flow(in_dims, flow_type=flow_type, flow_length=1, amortization=amortize)
        if (model in ['flow_p', 'flow_trans']):
            regression_model = FlowTransform(in_dims, blocks=blocks, flow_length=n_layers, amortize=amortize, eps_var=eps_var, var_type = var_type)
        elif (model in ['flow_m', 'flow_kl']):
            regression_model = FlowKL(in_dims, blocks=blocks, flow_length=n_layers, amortize=amortize, eps_var=eps_var, var_type = var_type)
        elif (model == 'flow_kl_f'):
            regression_model = FlowKLFull(in_dims, blocks=blocks, flow_length=n_layers, amortize=amortize, eps_var=eps_var, var_type = var_type)
        elif (model == 'flow_cde'):
            regression_model = FlowCDE(in_dims, blocks=blocks, flow_length=n_layers, amortize=amortize, eps_var=eps_var, var_type = var_type)
        elif (model == 'flow_ext'):
            regression_model = FlowExternal(in_dims, blocks=blocks, flow_length=n_layers, amortize=amortize, eps_var=eps_var, var_type = var_type)
        elif (model == 'flow_post'):
            regression_model = FlowPosterior(in_dims, blocks=blocks, flow_length=n_layers, amortize=amortize, eps_var=eps_var, var_type = var_type)
        elif (model == 'flow_dec'):
            regression_model = FlowDecoder(in_dims, blocks=blocks, flow_length=n_layers, amortize=amortize, eps_var=eps_var, var_type = var_type)
        else:
            raise ValueError('Invalid regressor choice : ' + model)
    else:
        raise ValueError('Invalid regressor choice : ' + model)
    return regression_model

class RegressionModel(nn.Module):
    
    def __init__(self, **kwargs):
        super(RegressionModel, self).__init__(**kwargs)
    
    def train_epoch(self, loader, loss, optimizer, args):
        self.train()
        full_loss = 0
        for (x, y, _, _) in loader:
            # Send to device
            x, y = x.to(args.device), y.to(args.device)
            optimizer.zero_grad()
            out = self(x)
            b_loss = loss(out, y)
            b_loss.backward()
            optimizer.step()
            full_loss += b_loss
        full_loss /= len(loader)
        return full_loss
    
    def eval_epoch(self, loader, loss, args):
        self.eval()
        full_loss = 0
        with torch.no_grad():
            for (x, y, _, _) in loader:
                x, y = x.to(args.device), y.to(args.device)
                out = self(x).data
                full_loss += loss(out, y)
            full_loss /= len(loader)
        return full_loss

class GatedMLP(RegressionModel):
    
    def __init__(self, in_size, out_size, hidden_size = 512, n_layers = 6, type_mod='gated', **kwargs):
        super(GatedMLP, self).__init__(**kwargs)
        dense_module = (type_mod == 'gated') and GatedDense or nn.Linear
        # Create modules
        modules = nn.Sequential()
        for l in range(n_layers):
            in_s = (l==0) and in_size or hidden_size
            out_s = (l == n_layers - 1) and out_size or hidden_size
            modules.add_module('l%i'%l, dense_module(in_s, out_s))
            if (l < n_layers - 1):
                modules.add_module('b%i'%l, nn.BatchNorm1d(out_s))
                modules.add_module('a%i'%l, nn.ReLU())
                modules.add_module('a%i'%l, nn.Dropout(p=.3))
        self.net = modules
    
    def init_parameters(self):
        """ Initialize internal parameters (sub-modules) """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
        
    def forward(self, inputs):
        # Flatten the input
        out = inputs.reshape(inputs.shape[0], -1)
        for m in range(len(self.net)):
            out = self.net[m](out)
        return out
    
class DecodeMLP(GatedMLP):
    
    def __init__(self, in_size, out_size, hidden_size = 512, n_layers = 6, type_mod='gated', **kwargs):
        super(DecodeMLP, self).__init__(in_size, np.prod(out_size), hidden_size, n_layers, type_mod, **kwargs)
        # Record final size
        self.out_size = out_size
        
    def forward(self, inputs):
        # Use super function
        out = GatedMLP.forward(self, inputs)
        # Reshape output
        out = out.reshape(inputs.shape[0], *self.out_size)
        return out
    
class GatedCNN(RegressionModel):
    
    def __init__(self, in_size, out_size, channels = 32, n_layers = 5, hidden_size = 512, n_mlp = 2, type_mod='gated', args=None):
        super(GatedCNN, self).__init__()
        conv_module = (type_mod == 'gated') and GatedConv2d or nn.Conv2d
        conv_module = (type_mod == 'residual') and ResConv2d or conv_module
        dense_module = (type_mod == 'gated') and GatedDense or nn.Linear
        # Create modules
        modules = nn.Sequential()
        size = [in_size[-2], in_size[-1]]
        in_channel = 1 if len(in_size)<3 else in_size[0] #in_size is (C,H,W) or (H,W)
        kernel = args.kernel
        stride = 2
        print(f"[DEBUG] GatedCNN init: in_size={in_size}, size={size}, in_channel={in_channel}, kernel={kernel}, stride={stride}")
        """ First do a CNN """
        for l in range(n_layers):
            dil = ((args.dilation == 3) and (2 ** l) or args.dilation)
            pad = 3 * (dil + 1)
            in_s = (l==0) and in_channel or channels
            out_s = (l == n_layers - 1) and 1 or channels
            modules.add_module('c2%i'%l, conv_module(in_s, out_s, kernel, stride, pad, dilation = dil))
            if (l < n_layers - 1):
                modules.add_module('b2%i'%l, nn.BatchNorm2d(out_s))
                modules.add_module('a2%i'%l, nn.ReLU())
                modules.add_module('d2%i'%l, nn.Dropout2d(p=.25))
            old_size = size.copy()
            size[0] = int((size[0]+2*pad-(dil*(kernel-1)+1))/stride+1)
            size[1] = int((size[1]+2*pad-(dil*(kernel-1)+1))/stride+1)
            print(f"[DEBUG] Layer {l}: dil={dil}, pad={pad}, {old_size} -> {size}")
        self.net = modules
        self.mlp = nn.Sequential()
        """ Then go through MLP """
        print(f"[DEBUG] Final CNN size: {size}, flattened: {size[0] * size[1]}")
        for l in range(n_mlp):
            in_s = (l==0) and (size[0] * size[1]) or hidden_size
            out_s = (l == n_mlp - 1) and out_size or hidden_size
            print(f"[DEBUG] MLP layer {l}: in_s={in_s}, out_s={out_s}")
            self.mlp.add_module('h%i'%l, dense_module(in_s, out_s))
            if (l < n_mlp - 1):
                self.mlp.add_module('b%i'%l, nn.BatchNorm1d(out_s))
                self.mlp.add_module('a%i'%l, nn.ReLU())
                self.mlp.add_module('d%i'%l, nn.Dropout(p=.25))
        self.cnn_size = size
        # Store attributes needed for MLP rebuilding
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.dense_module = dense_module
        self.n_mlp = n_mlp
    
    def init_parameters(self):
        """ Initialize internal parameters (sub-modules) """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
        
    def forward(self, inputs):
        print(f"[DEBUG] GatedCNN forward: input shape = {inputs.shape}")
        out = inputs.unsqueeze(1) if len(inputs.shape) < 4 else inputs # force to (batch, C, H, W)
        print(f"[DEBUG] GatedCNN forward: after unsqueeze shape = {out.shape}")
        for m in range(len(self.net)):
            out = self.net[m](out)
        print(f"[DEBUG] GatedCNN forward: after CNN shape = {out.shape}")
        out = out.reshape(inputs.shape[0], -1)
        print(f"[DEBUG] GatedCNN forward: after flatten shape = {out.shape}")
        
        # Handle variable input sizes by recalculating MLP if needed
        first_layer = self.mlp[0]
        expected_features = first_layer.h.in_features if hasattr(first_layer, 'h') else first_layer.in_features
        if out.shape[1] != expected_features:
            print(f"[DEBUG] MLP size mismatch: expected {expected_features}, got {out.shape[1]}")
            print(f"[DEBUG] Rebuilding MLP with correct input size: {out.shape[1]}")
            # Rebuild MLP with correct input size
            self._rebuild_mlp(out.shape[1])
            print(f"[DEBUG] MLP rebuilt successfully")
        
        for m in range(len(self.mlp)):
            out = self.mlp[m](out)
        return out
    
    def _rebuild_mlp(self, new_input_size):
        """ Rebuild MLP with new input size """
        print(f"[DEBUG] Rebuilding MLP with correct input size: {new_input_size}")
        
        # Rebuild MLP with correct input size
        new_mlp = nn.Sequential()
        for l in range(self.n_mlp):
            in_s = (l == 0) and new_input_size or self.hidden_size
            out_s = (l == self.n_mlp - 1) and self.out_size or self.hidden_size
            layer = self.dense_module(in_s, out_s)
            
            # Initialize weights properly to avoid NaN
            if hasattr(layer, 'weight'):
                nn.init.xavier_uniform_(layer.weight)
                if hasattr(layer, 'bias') and layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            
            new_mlp.add_module('h%i' % l, layer)
            if (l < self.n_mlp - 1):
                new_mlp.add_module('b%i' % l, nn.BatchNorm1d(out_s))
                new_mlp.add_module('a%i' % l, nn.ReLU())
                new_mlp.add_module('d%i' % l, nn.Dropout(p=.25))
        
        # Replace the old MLP
        self.mlp = new_mlp.to(next(self.parameters()).device)
        
        # Check for NaN in new weights
        for name, param in self.mlp.named_parameters():
            if torch.isnan(param).any():
                print(f"[WARNING] NaN detected in {name} after rebuild")
        
        print(f"[DEBUG] MLP rebuilt successfully")
    
class DecodeCNN(RegressionModel):
    
    def __init__(self, in_size, cnn_size, out_size, channels = 32, n_layers = 5, hidden_size = 512, n_mlp = 2, type_mod='gated', args=None):
        super(DecodeCNN, self).__init__()
        conv_module = (type_mod == 'gated') and GatedConvTranspose2d or nn.ConvTranspose2d
        conv_module = (type_mod == 'residual') and ResConvTranspose2d or conv_module
        dense_module = (type_mod == 'gated') and GatedDense or nn.Linear
        # Create modules
        self.cnn_size = [cnn_size[0], cnn_size[1]]
        size = cnn_size
        kernel = args.kernel
        stride = 2
        self.mlp = nn.Sequential()
        """ First go through MLP """
        for l in range(n_mlp):
            in_s = (l==0) and (in_size) or hidden_size
            out_s = (l == n_mlp - 1) and np.prod(cnn_size) or hidden_size
            self.mlp.add_module('h%i'%l, dense_module(in_s, out_s))
            if (l < n_mlp - 1):
                self.mlp.add_module('b%i'%l, nn.BatchNorm1d(out_s))
                self.mlp.add_module('a%i'%l, nn.ReLU())
                self.mlp.add_module('d%i'%l, nn.Dropout(p=.25))
        modules = nn.Sequential()
        """ Then do a CNN """
        for l in range(n_layers):
            dil = ((args.dilation == 3) and (2 ** ((n_layers - 1) - l)) or args.dilation)
            pad = 3 * (dil + 1)
            if (args.dilation == 1):
                pad = 2
            out_pad = (pad % 2)
            in_s = (l==0) and 1 or channels
            out_s = (l == n_layers - 1) and 1 or channels
            modules.add_module('c2%i'%l, conv_module(in_s, out_s, kernel, stride, pad, output_padding=out_pad, dilation = dil))
            if (l < n_layers - 1):
                modules.add_module('b2%i'%l, nn.BatchNorm2d(out_s))
                modules.add_module('a2%i'%l, nn.ReLU())
                modules.add_module('a2%i'%l, nn.Dropout2d(p=.25))
            size[0] = int((size[0] - 1) * stride - (2 * pad) + dil * (kernel - 1) + out_pad + 1)
            size[1] = int((size[1] - 1) * stride - (2 * pad) + dil * (kernel - 1) + out_pad + 1)
        self.net = modules
        self.out_size = out_size #(H,W) or (C,H,W)
    
    def init_parameters(self):
        """ Initialize internal parameters (sub-modules) """
        for param in self.parameters():
            param.data.uniform_(-0.01, 0.01)
    
    def _rebuild_mlp(self, new_input_size):
        """ Rebuild MLP with new input size """
        # Store the original MLP configuration
        n_mlp = len([name for name, module in self.mlp.named_modules() if 'h' in name])
        hidden_size = self.hidden_size
        out_size = self.out_size
        dense_module = self.dense_module
        
        # Rebuild MLP with correct input size
        new_mlp = nn.Sequential()
        for l in range(n_mlp):
            in_s = (l == 0) and new_input_size or hidden_size
            out_s = (l == n_mlp - 1) and out_size or hidden_size
            new_mlp.add_module('h%i' % l, dense_module(in_s, out_s))
            if (l < n_mlp - 1):
                new_mlp.add_module('b%i' % l, nn.BatchNorm1d(out_s))
                new_mlp.add_module('a%i' % l, nn.ReLU())
                new_mlp.add_module('d%i' % l, nn.Dropout(p=.25))
        
        # Replace the old MLP
        self.mlp = new_mlp.to(next(self.parameters()).device)
        
        # Initialize parameters for the new MLP
        for param in self.mlp.parameters():
            param.data.uniform_(-0.01, 0.01)
        
    def forward(self, inputs):
        out = inputs
        for m in range(len(self.mlp)):
            out = self.mlp[m](out)
        out = out.unsqueeze(1).reshape(-1, 1, self.cnn_size[0], self.cnn_size[1])
        for m in range(len(self.net)):
            out = self.net[m](out)
        
        # Handle variable output width using target_width if available
        if hasattr(self, 'target_width'):
            target_width = self.target_width
        else:
            target_width = self.out_size[-1] if len(self.out_size) >= 2 else self.out_size[0]
        
        # Get current dimensions (keeping channel dimension)
        # out shape: [batch, channel, height, width]
        current_height, current_width = out.shape[2], out.shape[3]
        
        # Determine target dimensions based on out_size format
        if len(self.out_size) == 2:  # (height, width)
            target_height, target_width = self.out_size[0], self.out_size[1]
        elif len(self.out_size) == 3:  # (channel, height, width)
            target_height, target_width = self.out_size[1], self.out_size[2]
        else:  # single dimension
            target_height, target_width = self.out_size[0], self.out_size[0]
        
        # Handle height dimension first
        if current_height != target_height:
            if current_height < target_height:
                # Pad height
                pad_height = target_height - current_height
                out = nn.functional.pad(out, (0, 0, 0, pad_height), mode='constant', value=0)
            else:
                # Crop height
                out = out[:, :, :target_height, :]
        
        # Handle width dimension
        current_width = out.shape[3]
        # Use the target_width parameter if provided, otherwise use out_size width
        if hasattr(self, 'target_width') and self.target_width is not None:
            final_target_width = self.target_width
        else:
            final_target_width = target_width
        if current_width != final_target_width:
            if current_width < final_target_width:
                # Pad width
                pad_width = final_target_width - current_width
                out = nn.functional.pad(out, (0, pad_width), mode='constant', value=0)
            else:
                # Crop width
                out = out[:, :, :, :final_target_width]
        
        # Keep channel dimension to match input format [batch, channel, height, width]
        # The input data has shape [batch, 1, height, width] so output should match
        # out = out.squeeze(1)  # [batch, height, width] - REMOVED
        # Keep as [batch, 1, height, width]
        
        return out
