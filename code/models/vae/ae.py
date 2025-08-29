import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.param_types import get_binary_param_indices

class RegressionAE(nn.Module):
    """
    Definition of AE model for all regression tasks.
    """
    
    def __init__(self, ae_model, latent_dims, regression_dims, recons_loss, regressor = None, regressor_name = '', **kwargs):
        super(RegressionAE, self).__init__(**kwargs)
        self.ae_model = ae_model
        self.recons_loss = recons_loss
        self.latent_dims = latent_dims
        self.regression_dims = regression_dims
        # Training stability parameters
        self.rec_loss_scale = 1e-4  # Scale down reconstruction loss to prevent numerical instability
        self.max_grad_norm = 1.0    # Gradient clipping threshold
        self.binary_indices = None
        if (regressor is None):
            self.regression_model = nn.Sequential(
                    nn.Linear(latent_dims, latent_dims * 2),
                    nn.ReLU(), nn.BatchNorm1d(latent_dims * 2),
                    nn.Linear(latent_dims * 2, latent_dims * 2),
                    nn.ReLU(), nn.BatchNorm1d(latent_dims * 2),
                    nn.Linear(latent_dims * 2, regression_dims)
                    # Removed ReLU and Hardtanh to allow full range of normalized parameter values
                    )
            regressor_name = 'mlp'
        else:
            self.regression_model = regressor
        self.regressor = regressor_name
        # Output activation to keep predictions in [0,1] with gradients
        self.output_activation = nn.Sigmoid() if self.regressor == 'mlp' else nn.Identity()
    
    def forward(self, x):
        # Auto-encode
        x_tilde, z_tilde, z_loss = self.ae_model(x)
        # Perform regression on params
        p_tilde = self.regression_model(z_tilde)
        p_tilde = self.output_activation(p_tilde)
        # Bound to valid parameter range [0, 1]
        p_tilde = torch.clamp(p_tilde, 0.0, 1.0)
        return p_tilde
    
    def train_epoch(self, loader, loss_params, optimizer, args):
        self.train()
        full_loss = 0
        # Cache binary indices lazily
        if self.binary_indices is None:
            try:
                self.binary_indices = get_binary_param_indices(list(loader.dataset.final_params))
            except Exception:
                self.binary_indices = []
        for (x, y, _, _) in loader:
            # Send to device
            x, y = x.to(args.device, non_blocking=True), y.to(args.device, non_blocking=True)
            # Auto-encode
            x_tilde, z_tilde, z_loss = self.ae_model(x)
            # Reconstruction loss with scaling to prevent numerical instability
            # Ensure tensors have compatible shapes for loss calculation
            x_flat = x.reshape(x.shape[0], -1)
            x_tilde_flat = x_tilde.reshape(x_tilde.shape[0], -1)
            rec_loss = self.recons_loss(x_tilde_flat, x_flat) / (x.shape[1] * x.shape[2])
            rec_loss = rec_loss * self.rec_loss_scale  # Scale down reconstruction loss
            if (self.regressor == 'mlp'):
                # Perform regression on params
                p_tilde = self.regression_model(z_tilde)
                # Output activation + bound predictions to [0, 1]
                p_tilde = self.output_activation(p_tilde)
                p_tilde = torch.clamp(p_tilde, 0.0, 1.0)
                # Regression loss (optionally weighted by parameter variability)
                if isinstance(loss_params, nn.SmoothL1Loss) and hasattr(loader.dataset, 'final_std'):
                    with torch.no_grad():
                        w = loader.dataset.final_std.to(y.device)
                        # Avoid zero weights; emphasize informative parameters
                        w = w / (w.mean() + 1e-8)
                        w = torch.clamp(w, min=1e-3)
                    reg_loss = F.smooth_l1_loss(p_tilde, y, reduction='none')
                    reg_loss = (reg_loss * w.unsqueeze(0)).mean()
                else:
                    reg_loss = loss_params(p_tilde, y)
                # Add BCE for binary toggles
                if self.binary_indices:
                    bidx = torch.tensor(self.binary_indices, device=y.device, dtype=torch.long)
                    bce = F.binary_cross_entropy(p_tilde[:, bidx], y[:, bidx])
                    reg_loss = reg_loss + bce
            else:
                # Use log probability model
                p_tilde, reg_loss = self.regression_model.log_prob(z_tilde, y)
            # Final loss
            b_loss = rec_loss + (args.beta * z_loss) + (args.gamma * reg_loss)
            
            # Check for NaN/Inf before backward pass
            if torch.isnan(b_loss) or torch.isinf(b_loss):
                print(f"Warning: NaN/Inf detected in loss, skipping batch")
                continue
            
            # Perform backward
            optimizer.zero_grad()
            b_loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
            
            # Check for NaN gradients
            has_nan_grad = False
            for param in self.parameters():
                if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                    has_nan_grad = True
                    break
            
            if has_nan_grad:
                print(f"Warning: NaN/Inf gradients detected, skipping optimizer step")
                continue
            
            optimizer.step()
            full_loss += b_loss.detach()
        full_loss /= len(loader)
        return full_loss
    
    def eval_epoch(self, loader, loss_params, args):
        self.eval()
        full_loss = 0
        valid_batches = 0
        # Cache binary indices lazily
        if self.binary_indices is None:
            try:
                self.binary_indices = get_binary_param_indices(list(loader.dataset.final_params))
            except Exception:
                self.binary_indices = []
        with torch.no_grad():
            for (x, y, _, _) in loader:
                x, y = x.to(args.device, non_blocking=True), y.to(args.device, non_blocking=True)
                # Auto-encode
                x_tilde, z_tilde, z_loss = self.ae_model(x)
                # Perform regression on params
                p_tilde = self.regression_model(z_tilde)
                # Output activation + bound predictions to [0, 1]
                p_tilde = self.output_activation(p_tilde)
                p_tilde = torch.clamp(p_tilde, 0.0, 1.0)
                # Regression loss (optionally weighted)
                if isinstance(loss_params, nn.SmoothL1Loss) and hasattr(loader.dataset, 'final_std'):
                    with torch.no_grad():
                        w = loader.dataset.final_std.to(y.device)
                        w = w / (w.mean() + 1e-8)
                        w = torch.clamp(w, min=1e-3)
                    reg_loss = F.smooth_l1_loss(p_tilde, y, reduction='none')
                    reg_loss = (reg_loss * w.unsqueeze(0)).mean()
                else:
                    reg_loss = loss_params(p_tilde, y)
                # Add BCE for binary toggles
                if self.binary_indices:
                    bidx = torch.tensor(self.binary_indices, device=y.device, dtype=torch.long)
                    bce = F.binary_cross_entropy(p_tilde[:, bidx], y[:, bidx])
                    reg_loss = reg_loss + bce
                
                # Check for NaN/Inf in evaluation loss
                if torch.isnan(reg_loss) or torch.isinf(reg_loss):
                    print(f"Warning: NaN/Inf detected in eval loss, skipping batch")
                    continue
                
                full_loss += reg_loss
                valid_batches += 1
            
            if valid_batches > 0:
                full_loss /= valid_batches
            else:
                print(f"Warning: No valid batches in evaluation, returning NaN")
                full_loss = torch.tensor(float('nan'))
        return full_loss

class DisentanglingAE(RegressionAE):
    """
    Definition of regression AE model with the added 
    """
    
    def __init__(self, ae_model, latent_dims, regression_dims, recons_loss, regressor = None, regressor_name = '', disentangling = None, semantic_dim = -1, **kwargs):
        super(DisentanglingAE, self).__init__(ae_model, latent_dims, regression_dims, recons_loss, regressor, regressor_name)
        # Disentangling model
        self.disentangling = disentangling
        # Semantic dim to evaluate
        self.semantic_dim = semantic_dim
    
    def forward(self, x):
        # Auto-encode
        #x_tilde, z_tilde, z_loss = self.ae_model(x)
        # Disentangling part
        #z_tilde, _ = self.disentangling(z_tilde)
        # Encode the inputs
        z_params = self.ae_model.encode(x)
        # Obtain latent samples and latent loss
        z_tilde, z_loss = self.ae_model.latent(x, z_params)
        # Perform disentangling
        z_tilde, _ = self.disentangling(z_tilde)
        # Decode the samples
        #x_tilde = self.decode(z_tilde)
        # Perform regression on params
        p_tilde = self.regression_model(z_tilde)
        p_tilde = self.output_activation(p_tilde)
        # Bound to valid parameter range [0, 1]
        p_tilde = torch.clamp(p_tilde, 0.0, 1.0)
        return p_tilde
    
    def train_epoch(self, loader, loss_params, optimizer, args):
        self.train()
        full_loss = 0
        for (x, y, meta, _) in loader:
            # Send to device
            x, y, meta = x.to(args.device, non_blocking=True), y.to(args.device, non_blocking=True), meta.to(args.device, non_blocking=True)
            # Extract current meta-tag
            meta = meta[:, self.semantic_dim].squeeze(1)
            target = meta[:, 1].long()
            # Separate examples
            loss_mask = 1 - meta[:, 2]
            observed = loss_mask.eq(1)
            unknown = loss_mask.eq(0)
            # Auto-encode
            #x_tilde, z_tilde, z_loss = self.ae_model(x)
            # Reconstruction loss
            #rec_loss = self.recons_loss(x_tilde, x) / (x.shape[1] * x.shape[2])
            # Disentangling part
            #z_tilde, dis_loss = self.disentangling(z_tilde, (meta, target, observed, unknown))            
            # Encode the inputs
            z_params = self.ae_model.encode(x)
            # Obtain latent samples and latent loss
            z_tilde, z_loss = self.ae_model.latent(x, z_params)
            # Disentangling part
            z_tilde, dis_loss = self.disentangling(z_tilde, (meta, target, observed, unknown))            
            # Decode the samples
            x_tilde = self.ae_model.decode(z_tilde)
            # Reconstruction loss with scaling to prevent numerical instability
            # Ensure tensors have compatible shapes for loss calculation
            x_flat = x.reshape(x.shape[0], -1)
            x_tilde_flat = x_tilde.reshape(x_tilde.shape[0], -1)
            rec_loss = self.recons_loss(x_tilde_flat, x_flat) / (x.shape[1] * x.shape[2])
            rec_loss = rec_loss * self.rec_loss_scale  # Scale down reconstruction loss
            # Regression part
            if (self.regressor == 'mlp'):
                # Perform regression on params
                p_tilde = self.regression_model(z_tilde)
                p_tilde = self.output_activation(p_tilde)
                # Bound predictions to [0, 1]
                p_tilde = torch.clamp(p_tilde, 0.0, 1.0)
                # Regression loss (optionally weighted)
                if isinstance(loss_params, nn.SmoothL1Loss) and hasattr(loader.dataset, 'final_std'):
                    with torch.no_grad():
                        w = loader.dataset.final_std.to(y.device)
                        w = w / (w.mean() + 1e-8)
                        w = torch.clamp(w, min=1e-3)
                    reg_loss = F.smooth_l1_loss(p_tilde, y, reduction='none')
                    reg_loss = (reg_loss * w.unsqueeze(0)).mean()
                else:
                    reg_loss = loss_params(p_tilde, y)
            else:
                # Use log probability model
                p_tilde, reg_loss = self.regression_model.log_prob(z_tilde, y)
            # Final loss
            b_loss = rec_loss + (args.beta * z_loss) + (args.gamma * reg_loss) + (args.beta * dis_loss)
            
            # Check for NaN/Inf before backward pass
            if torch.isnan(b_loss) or torch.isinf(b_loss):
                print(f"Warning: NaN/Inf detected in loss, skipping batch")
                continue
            
            # Perform backward
            optimizer.zero_grad()
            b_loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
            
            # Check for NaN gradients
            has_nan_grad = False
            for param in self.parameters():
                if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                    has_nan_grad = True
                    break
            
            if has_nan_grad:
                print(f"Warning: NaN/Inf gradients detected, skipping optimizer step")
                continue
            
            optimizer.step()
            full_loss += b_loss.detach()
        full_loss /= len(loader)
        return full_loss
    
    def eval_epoch(self, loader, loss_params, args):
        self.eval()
        full_loss = 0
        valid_batches = 0
        with torch.no_grad():
            for (x, y, meta, _) in loader:
                # Send to device
                x, y, meta = x.to(args.device, non_blocking=True), y.to(args.device, non_blocking=True), meta.to(args.device, non_blocking=True)
                # Extract current meta-tag
                #meta = meta[:, self.semantic_dim].squeeze(1)
                #target = meta[:, 1].long()
                # Separate examples
                #loss_mask = 1 - meta[:, 2]
                #observed_examples = loss_mask.eq(1)
                #unknown_examples = loss_mask.eq(0)
                # Auto-encode
                #x_tilde, z_tilde, z_loss = self.ae_model(x)
                # Disentangling part
                #z_tilde, _ = self.disentangling(z_tilde)
                # Encode the inputs
                z_params = self.ae_model.encode(x)
                # Obtain latent samples and latent loss
                z_tilde, z_loss = self.ae_model.latent(x, z_params)
                # Disentangling part
                z_tilde, _ = self.disentangling(z_tilde)            
                # Perform regression on params
                p_tilde = self.regression_model(z_tilde)
                p_tilde = self.output_activation(p_tilde)
                # Bound predictions to [0, 1]
                p_tilde = torch.clamp(p_tilde, 0.0, 1.0)
                # Regression loss (optionally weighted)
                if isinstance(loss_params, nn.SmoothL1Loss) and hasattr(loader.dataset, 'final_std'):
                    with torch.no_grad():
                        w = loader.dataset.final_std.to(y.device)
                        w = w / (w.mean() + 1e-8)
                        w = torch.clamp(w, min=1e-3)
                    reg_loss = F.smooth_l1_loss(p_tilde, y, reduction='none')
                    reg_loss = (reg_loss * w.unsqueeze(0)).mean()
                else:
                    reg_loss = loss_params(p_tilde, y)
                
                # Check for NaN/Inf in evaluation loss
                if torch.isnan(reg_loss) or torch.isinf(reg_loss):
                    print(f"Warning: NaN/Inf detected in eval loss, skipping batch")
                    continue
                
                # Compute full loss
                full_loss += reg_loss
                valid_batches += 1
            
            if valid_batches > 0:
                full_loss /= valid_batches
            else:
                print(f"Warning: No valid batches in evaluation, returning NaN")
                full_loss = torch.tensor(float('nan'))
        return full_loss
    
class AE(nn.Module):
    
    def __init__(self, encoder, decoder, encoder_dims, latent_dims):
        super(AE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dims = latent_dims
        self.map_latent = nn.Linear(encoder_dims, latent_dims)
        self.apply(self.init_parameters)
    
    def init_parameters(self, m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        
    def encode(self, x):
        x = self.encoder(x)
        return x
    
    def decode(self, z):
        return self.decoder(z)
    
    def regularize(self, z):
        z = self.map_latent(z)
        return z, torch.zeros(z.shape[0]).to(z.device).mean()

    def forward(self, x):
        # Encode the inputs
        z = self.encode(x)
        # Potential regularization
        z_tilde, z_loss = self.regularize(z)
        # Decode the samples
        x_tilde = self.decode(z_tilde)
        return x_tilde, z_tilde, z_loss
