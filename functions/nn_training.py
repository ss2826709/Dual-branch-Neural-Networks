import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import copy

# ── Seed ─────────────────────────────────────────────────────────────────────
def seed_everything(seed=88):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# ── Logging ───────────────────────────────────────────────────────────────────
def track_loss(epoch, epochs, avg_loss):
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1:3d}/{epochs}] | Train Loss: {avg_loss:.4f}")

# ── Loss functions ────────────────────────────────────────────────────────────
def cosine_diversity_loss(s_delta):
    """
    Penalize cosine similarity between all pairs of source correction vectors.
    s_delta : [B, K] or [B, T, K]  — K = number of sources
    Returns a scalar loss.
    """
    K = s_delta.shape[-1]
    d = s_delta.reshape(-1, K)       # [B*T, K] — one column per source
    d = F.normalize(d, dim=0)        # unit-norm each source vector (column-wise)
    sim_matrix = d.T @ d             # [K, K] pairwise cosine similarities
    mask = ~torch.eye(K, dtype=torch.bool, device=s_delta.device)
    return sim_matrix[mask].abs().mean()

def twostage_loss(y_pred, yb, s_delta, lambda_sim):
    loss_main = F.mse_loss(y_pred, yb)
    loss_sim  = cosine_diversity_loss(s_delta)
    return loss_main + lambda_sim * loss_sim

# ── Early stopping helper ─────────────────────────────────────────────────────
def update_early_stopping(val_loss, best_val, best_epoch,
                           patience_counter, epoch, patience):
    """
    Update early-stopping state for one epoch.
    Returns (best_val, best_epoch, patience_counter, should_stop).
    best_epoch is 1-indexed so it can be passed directly as 'epochs'
    to the training functions for the final run.
    """
    
    if val_loss < best_val:
        best_val         = val_loss
        best_epoch       = epoch + 1   # 1-indexed
        patience_counter = 0
        should_stop      = False
    else:
        patience_counter += 1
        should_stop       = patience_counter >= patience
        
    return best_val, best_epoch, patience_counter, should_stop

# ── Core epoch helpers ────────────────────────────────────────────────────────
def _one_epoch_loader(model, loader, loss_fn, optimizer):

    """
    One training epoch using per-batch gradient updates (standard mini-batch SGD).

    loss_fn(model_output, yb) -> scalar loss tensor
        Standard models:    loss_fn = criterion
        Two-stage models:   loss_fn unpacks the 4-tuple output (y_pred, _, s_delta, _) internally

    Returns the average loss across all batches.
    """
    
    model.train()

    total_loss = 0.0
    total_samples = 0

    for xb, yb in loader:
        optimizer.zero_grad()

        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()

        batch_size     = xb.size(0)
        total_loss    += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples

def _one_epoch_tensors(model, tensors, group_keys, loss_fn, optimizer):
    
    """
    tensors[T] must be either:
        (xb, yb)                for single-input models (CNN1D)
        (xb_ts, xb_static, yb) for DB models
    The last element is always yb; everything before it is forwarded to the model.
    loss_fn(model_output, yb) -> scalar loss tensor (same convention as above)

    """
    model.train()
    total_loss = 0.0
    total_samples = 0
    random.shuffle(group_keys)              # vary processing order each epoch

    for T in group_keys:
        optimizer.zero_grad()
        *inputs, yb = tensors[T]             
        loss = loss_fn(model(*inputs), yb)
        loss.backward()
        optimizer.step()
        
        batch_size = yb.numel()
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        
    return total_loss / total_samples

# ── Public training functions ─────────────────────────────────────────────────
#
# Dual-mode design
# ────────────────
# With validation (val_X/val_y or vl_tensors + patience):
#   → early stopping, returns (train_losses, best_val, best_epoch)
#   → used in hp_tuning: best_epoch is aggregated across folds to set final epochs
#
# Without validation (defaults None):
#   → fixed-epoch training, returns train_losses only
#   → used in models_for_eval: call signatures are unchanged from the original
#
# Loader-based  (per-batch updates) : train_MLP, train_TwostageMLP
# Tensors-based (per-group updates) : train_CNN1D, train_DB,
#                                     train_TwostageCNN1D, train_TwostageDB

def train_MLP(model, loader, criterion, optimizer, epochs,
              val_X=None, val_y=None, patience=None):
    """
    Standard mini-batch SGD training for MLP.

    Validation mode  (val_X, val_y, patience provided):
        Runs early stopping; returns (train_losses, best_val, best_epoch).

    Fixed-epoch mode (no validation args):
        Trains for 'epochs' epochs; returns train_losses.
    """
    use_validation = (val_X is not None and val_y is not None)

    if use_validation and patience is None:
        raise ValueError("patience must be provided when val_X and val_y are given")

    train_losses     = []
    val_losses       = []
    best_val         = float("inf")
    best_epoch       = 0
    best_state       = None 
    patience_counter = 0

    for epoch in range(epochs):
        avg_loss = _one_epoch_loader(model, loader, criterion, optimizer)
        train_losses.append(avg_loss)

        if not use_validation:
            track_loss(epoch, epochs, avg_loss)
            continue

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(val_X), val_y).item()
            val_losses.append(val_loss)

        if val_loss < best_val:
            best_state = copy.deepcopy(model.state_dict())  

        best_val, best_epoch, patience_counter, stop = update_early_stopping(
            val_loss, best_val, best_epoch, patience_counter, epoch, patience)
        if stop:
            break

    if use_validation:
        if best_state is not None:
            model.load_state_dict(best_state)  
        return train_losses, best_val, best_epoch, val_losses
    return train_losses


def train_TwostageMLP(model, loader, optimizer, epochs, lambda_sim=0.05,
                      val_X=None, val_y=None, patience=None):
    """
    Two-stage MLP with cosine diversity loss (per-batch updates).

    Validation mode  (val_X, val_y, patience provided):
        Runs early stopping; returns (train_losses, best_val, best_epoch).

    Fixed-epoch mode (no validation args):
        Trains for 'epochs' epochs; returns train_losses.
    """
    def loss_fn(outputs, yb):
        y_pred, _, s_delta, _ = outputs
        return twostage_loss(y_pred, yb, s_delta, lambda_sim)

    use_validation = (val_X is not None and val_y is not None)

    if use_validation and patience is None:
        raise ValueError("patience must be provided when val_X and val_y are given")

    train_losses     = []
    val_losses       = []
    best_val         = float("inf")
    best_epoch       = 0
    best_state       = None 
    patience_counter = 0

    for epoch in range(epochs):
        avg_loss = _one_epoch_loader(model, loader, loss_fn, optimizer)
        train_losses.append(avg_loss)

        if not use_validation:
            track_loss(epoch, epochs, avg_loss)
            continue

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(val_X), val_y).item()
            val_losses.append(val_loss)

        if val_loss < best_val:
            best_state = copy.deepcopy(model.state_dict())  

        best_val, best_epoch, patience_counter, stop = update_early_stopping(
            val_loss, best_val, best_epoch, patience_counter, epoch, patience)
        if stop:
            break
            
    if use_validation:
        if best_state is not None:
            model.load_state_dict(best_state)  
        return train_losses, best_val, best_epoch, val_losses
    return train_losses


def train_CNN1D(model, tensors, criterion, optimizer, epochs,
                vl_tensors=None, patience=None):
    """
    Per-group gradient updates for CNN1D.
    tensors[T] = (X_t, y_t)

    Validation mode  (vl_tensors + patience provided):
        Computes val loss by concatenating predictions across all validation groups;
        returns (train_losses, best_val, best_epoch).

    Fixed-epoch mode (no validation args):
        Trains for 'epochs' epochs; returns train_losses.
    """
    use_validation = (vl_tensors is not None)

    if use_validation and patience is None:
        raise ValueError("patience must be provided when vl_tensors is given")

    group_keys     = list(tensors.keys())

    train_losses     = []
    val_losses       = []
    best_val         = float("inf")
    best_epoch       = 0
    best_state       = None 
    patience_counter = 0

    for epoch in range(epochs):
        epoch_loss = _one_epoch_tensors(model, tensors, group_keys,
                                        criterion, optimizer)
        train_losses.append(epoch_loss)

        if not use_validation:
            track_loss(epoch, epochs, epoch_loss)
            continue

        model.eval()
        with torch.no_grad():
            val_preds, val_targets = [], []  
            for X_t, y_t in vl_tensors.values():  
                y_pred = model(X_t).reshape_as(y_t)  
                mask = torch.isfinite(y_t)  
                if mask.any():  
                    val_preds.append(y_pred[mask])   
                    val_targets.append(y_t[mask])    
            if not val_targets:  
                raise ValueError("No observed validation targets are available for masked validation loss.")  
            val_loss = criterion(torch.cat(val_preds), torch.cat(val_targets)).item()  
            val_losses.append(val_loss)

        if val_loss < best_val:
            best_state = copy.deepcopy(model.state_dict())  

        best_val, best_epoch, patience_counter, stop = update_early_stopping(
            val_loss, best_val, best_epoch, patience_counter, epoch, patience)
        if stop:
            break

    if use_validation:
        if best_state is not None:
            model.load_state_dict(best_state)  
        return train_losses, best_val, best_epoch, val_losses
    return train_losses


def train_DB(model, tensors, criterion, optimizer, epochs,
                     vl_tensors=None, patience=None):
    """
    Per-group gradient updates for DB (CNN1D + static MLP branches).
    tensors[T] = (X_ts, X_static, y_t)

    Validation mode  (vl_tensors + patience provided):
        returns (train_losses, best_val, best_epoch).

    Fixed-epoch mode (no validation args):
        returns train_losses.
    """
    use_validation = (vl_tensors is not None)

    if use_validation and patience is None:
        raise ValueError("patience must be provided when vl_tensors is given")

    group_keys     = list(tensors.keys())

    train_losses     = []
    val_losses       = []
    best_val         = float("inf")
    best_epoch       = 0
    best_state       = None 
    patience_counter = 0

    for epoch in range(epochs):
        epoch_loss = _one_epoch_tensors(model, tensors, group_keys,
                                        criterion, optimizer)
        train_losses.append(epoch_loss)

        if not use_validation:
            track_loss(epoch, epochs, epoch_loss)
            continue

        model.eval()
        with torch.no_grad():
            val_preds, val_targets = [], [] 
            for X_ts, X_static, y_t in vl_tensors.values():  
                y_pred = model(X_ts, X_static).reshape_as(y_t)  
                mask = torch.isfinite(y_t) 
                if mask.any():  
                    val_preds.append(y_pred[mask])  
                    val_targets.append(y_t[mask])  
            if not val_targets:  
                raise ValueError("No observed validation targets are available for masked validation loss.")  
            val_loss = criterion(torch.cat(val_preds), torch.cat(val_targets)).item()  
            val_losses.append(val_loss)

        if val_loss < best_val:
            best_state = copy.deepcopy(model.state_dict())  

        best_val, best_epoch, patience_counter, stop = update_early_stopping(
            val_loss, best_val, best_epoch, patience_counter, epoch, patience)
        
        if stop:
            break

    if use_validation:
        if best_state is not None:
            model.load_state_dict(best_state)  
        return train_losses, best_val, best_epoch, val_losses
    return train_losses


def train_TwostageCNN1D(model, tensors, optimizer, epochs, lambda_sim=0.05,
                         vl_tensors=None, patience=None):
    """
    Two-stage CNN1D with cosine diversity loss (per-group updates).
    tensors[T] = (X_t, y_t)

    Validation mode  (vl_tensors + patience provided):
        Val loss = twostage_loss (MSE + diversity) for consistency with training.
        returns (train_losses, best_val, best_epoch).

    Fixed-epoch mode (no validation args):
        returns train_losses.
    """
    def loss_fn(outputs, yb):
        y_pred, _, s_delta, _ = outputs
        return twostage_loss(y_pred, yb, s_delta, lambda_sim)

    use_validation = (vl_tensors is not None)

    if use_validation and patience is None:
        raise ValueError("patience must be provided when vl_tensors is given")

    group_keys     = list(tensors.keys())

    train_losses     = []
    val_losses       = []
    best_val         = float("inf")
    best_epoch       = 0
    best_state       = None 
    patience_counter = 0

    for epoch in range(epochs):
        epoch_loss = _one_epoch_tensors(model, tensors, group_keys, 
                                        loss_fn, optimizer)
        train_losses.append(epoch_loss)

        if not use_validation:
            track_loss(epoch, epochs, epoch_loss)
            continue

        model.eval()
        with torch.no_grad():
            val_preds, val_targets, val_s_delta = [], [], []  
            for X_t, y_t in vl_tensors.values():  
                y_pred, _, s_delta, _ = model(X_t)
                mask = torch.isfinite(y_t)   
                if mask.any():   
                    val_preds.append(y_pred[mask])   
                    val_targets.append(y_t[mask])    
                    val_s_delta.append(s_delta[mask])   
            if not val_targets:  
                raise ValueError("No observed validation targets are available for masked validation loss.")  
            val_loss = twostage_loss(torch.cat(val_preds),
                                     torch.cat(val_targets),
                                     torch.cat(val_s_delta, dim=0),
                                     lambda_sim).item() 
            val_losses.append(val_loss)

        if val_loss < best_val:
            best_state = copy.deepcopy(model.state_dict())  

        best_val, best_epoch, patience_counter, stop = update_early_stopping(
            val_loss, best_val, best_epoch, patience_counter, epoch, patience)
        if stop:
            break

    if use_validation:
        if best_state is not None:
            model.load_state_dict(best_state) 
        return train_losses, best_val, best_epoch, val_losses
    return train_losses


def train_TwostageDB(model, tensors, optimizer, epochs, lambda_sim=0.05,
                              vl_tensors=None, patience=None):
    """
    Two-stage DB with cosine diversity loss (per-group updates).
    tensors[T] = (X_ts, X_static, y_t)

    Validation mode  (vl_tensors + patience provided):
        Val loss = twostage_loss (MSE + diversity).
        returns (train_losses, best_val, best_epoch).

    Fixed-epoch mode (no validation args):
        returns train_losses.
    """
    def loss_fn(outputs, yb):
        y_pred, _, s_delta, _ = outputs
        return twostage_loss(y_pred, yb, s_delta, lambda_sim)

    use_validation = (vl_tensors is not None)

    if use_validation and patience is None:
        raise ValueError("patience must be provided when vl_tensors is given")

    group_keys     = list(tensors.keys())

    train_losses     = []
    val_losses       = []
    best_val         = float("inf")
    best_epoch       = 0
    best_state       = None
    patience_counter = 0

    for epoch in range(epochs):
        epoch_loss = _one_epoch_tensors(model, tensors, group_keys,
                                        loss_fn, optimizer)
        train_losses.append(epoch_loss)

        if not use_validation:
            track_loss(epoch, epochs, epoch_loss)
            continue

        model.eval()
        with torch.no_grad():
            val_preds, val_targets, val_s_delta = [], [], []  
            for X_ts, X_static, y_t in vl_tensors.values():   
                y_pred, _, s_delta, _ = model(X_ts, X_static)
                mask = torch.isfinite(y_t)  
                if mask.any():  
                    val_preds.append(y_pred[mask])     
                    val_targets.append(y_t[mask])      
                    val_s_delta.append(s_delta[mask]) 
            if not val_targets: 
                raise ValueError("No observed validation targets are available for masked validation loss.")  #FIX: 
            val_loss = twostage_loss(torch.cat(val_preds),
                                     torch.cat(val_targets),
                                     torch.cat(val_s_delta, dim=0),
                                     lambda_sim).item()  
            val_losses.append(val_loss)

        if val_loss < best_val:
            best_state = copy.deepcopy(model.state_dict()) 

        best_val, best_epoch, patience_counter, stop = update_early_stopping(
            val_loss, best_val, best_epoch, patience_counter, epoch, patience)
        if stop:
            break

    if use_validation:
        if best_state is not None:
            model.load_state_dict(best_state)  
        return train_losses, best_val, best_epoch, val_losses
    return train_losses
