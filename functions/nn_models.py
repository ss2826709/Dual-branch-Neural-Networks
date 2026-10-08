"""Neural network architectures for PM2.5 predictions.

Single-stage models
-------------------
MLP                : Multilayer perceptron. 
CNN1D              : 1-D causal convolutional network.
DB                 : Parallel CNN1D + MLP branches.

Two-stage models
----------------
TwoStageMLP        : Shared MLP backbone   + per-source correction heads.
TwoStageCNN1D      : Shared CNN1D backbone + per-source correction heads.
TwoStageDB         : Shared DB backbone    + per-source correction heads.

Two-stage component
----------------
Stage 1 — A shared network learns a hidden representation from all input features.
Stage 2 — Each correction head receives the hidden representation and the corresponding CMAQ source contribution, then predicts a residual correction (s_delta). 
Corrected source contributions are summed to produce the final PM2.5 prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _optional_tensor_buffer(module, name, value, shape):  
    if value is None:                                     
        module.register_buffer(name, None)                
    else:                                                                                       
        module.register_buffer(name, torch.as_tensor(value, dtype=torch.float32).view(*shape))  


def _sum_corrected_sources_in_target_scale(s_corrected, source_mean, source_scale, target_mean, target_scale):  
    if source_mean is None or source_scale is None or target_mean is None or target_scale is None:              
        return s_corrected.sum(dim=-1, keepdim=(s_corrected.ndim == 2))                                        
    s_corrected_raw = s_corrected * source_scale + source_mean                                                  
    y_pred_raw = s_corrected_raw.sum(dim=-1, keepdim=(s_corrected.ndim == 2))                                  
    return (y_pred_raw - target_mean) / target_scale                                                            


def suggest_chain(trial, names, low=5, high=8):
    """Suggest a non-increasing chain of power-of-two widths and stash the
    resolved values as user attrs under `names`."""
    widths = []
    cur_high = high
    for name in names:
        exp = trial.suggest_int(f"{name}_exp", low, cur_high)
        w = 2 ** exp
        trial.set_user_attr(name, w)
        widths.append(w)
        cur_high = exp
    return widths


# =============================================================================
# Single-stage models
# =============================================================================

class MLP(nn.Module):

    """Three-hidden-layer MLP.

    Architecture:  Linear → ReLU → Dropout  (× 3)  → Linear output

    Parameters
    ----------
    input_size   : Number of input features.
    h1, h2, hout : Widths of hidden layers 1, 2, and 3.
    drop_rate    : Dropout probability applied after each hidden layer.
    output_size  : Number of output neurons (default 1 for scalar regression).

    Input  : x — [B, input_size]
    Output :   — [B, output_size]
    """

    def __init__(self, input_size, h1, h2, hout, drop_rate, output_size=1):
        super().__init__()
        self.fc1      = nn.Linear(input_size, h1)
        self.dropout1 = nn.Dropout(drop_rate)
        self.fc2      = nn.Linear(h1, h2)
        self.dropout2 = nn.Dropout(drop_rate)
        self.fc3      = nn.Linear(h2, hout)
        self.dropout3 = nn.Dropout(drop_rate)
        self.out      = nn.Linear(hout, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # [B, h1]
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))  # [B, h2]
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))  # [B, hout]    
        x = self.dropout3(x)
        preds = self.out(x)      # [B, output_size]    
        
        return preds


class CNN1D(nn.Module):

    """Three-layer 1-D causal CNN.

    Left-only (causal) zero-padding is applied before each convolution so the sequence length T is preserved and no future information leaks into past steps.

    Architecture:  CausalPad → Conv1d → ReLU → Dropout  (× 3)  → Conv1d head

    Parameters
    ----------
    in_features  : Number of input channels (features per timestep).
    h1, h2, hout : Channel widths of Conv layers 1, 2, and 3.
    drop_rate    : Dropout probability applied after each conv layer.

    Input  : x — [B, T, in_features]
    Output :   — [B, T]
    """

    def __init__(self, in_features, h1, h2, hout, drop_rate):    
        super().__init__()
        self.conv1    = nn.Conv1d(in_features, h1,   kernel_size=7, padding=0)
        self.dropout1 = nn.Dropout(drop_rate)
        self.conv2    = nn.Conv1d(h1,          h2,   kernel_size=5, padding=0)
        self.dropout2 = nn.Dropout(drop_rate)
        self.conv3    = nn.Conv1d(h2,          hout, kernel_size=3, padding=0)
        self.dropout3 = nn.Dropout(drop_rate)
        self.head     = nn.Conv1d(hout, 1, kernel_size=1)

    def forward(self, x):
        x = x.permute(0, 2, 1)              # [B, T, F] → [B, F, T]
        x = F.pad(x, (6, 0), value=0)       # causal pad → [B, F,  T+6] 
        x = F.relu(self.conv1(x))           # → [B, h1, T] 
        x = self.dropout1(x)
        x = F.pad(x, (4, 0), value=0)       # causal pad → [B, h1, T+4]  
        x = F.relu(self.conv2(x))           # → [B, h2, T]
        x = self.dropout2(x)
        x = F.pad(x, (2, 0), value=0)       # causal pad → [B, h2, T+2]
        x = F.relu(self.conv3(x))           # → [B, hout, T]
        x = self.dropout3(x)
        x = self.head(x)                    # → [B, 1, T]
        return x.squeeze(1)                 # → [B, T]

class DB(nn.Module):

    """DB model with parallel CNN1D and MLP branches.

    The CNN1D branch processes time-series features 
    The MLP branch processes static features.  
    Their outputs are concatenated channel-wise and passed through a 1×1 Conv1d fusion head to
    produce a per-timestep scalar prediction.

    Parameters
    ----------
    n_fts            : Number of time-series input channels.
    n_fstatic        : Number of static input features.
    hts1/hts2/hts3   : CNN1D hidden channel widths for layers 1, 2, and 3.
    hstt1/hstt2      : MLP hidden widths for layers 1 and 2.
    drop_rate_ts     : Dropout probability for the CNN1D branch.
    drop_rate_stt    : Dropout probability for the MLP branch.

    Inputs  : x_ts     — [B, T, n_fts]
              x_static — [B, n_fstatic]
    Output  :          — [B, T]
    """

    def __init__(self, n_fts, n_fstatic, hts1, hts2, hts3, hstt1, hstt2, drop_rate_ts, drop_rate_stt):
        super().__init__()

        # ── CNN1D branch (time-series features) ──────────────────────────────
        self.conv1       = nn.Conv1d(n_fts,  hts1, kernel_size=7, padding=0)
        self.dropout_ts1 = nn.Dropout(drop_rate_ts)
        self.conv2       = nn.Conv1d(hts1,  hts2, kernel_size=5, padding=0)
        self.dropout_ts2 = nn.Dropout(drop_rate_ts)
        self.conv3       = nn.Conv1d(hts2,  hts3, kernel_size=3, padding=0)
        self.dropout_ts3 = nn.Dropout(drop_rate_ts)

        # ── MLP branch (static features) ─────────────────────────────────────
        self.fc1          = nn.Linear(n_fstatic, hstt1)
        self.dropout_stt1 = nn.Dropout(drop_rate_stt)
        self.fc2          = nn.Linear(hstt1, hstt2)
        self.dropout_stt2 = nn.Dropout(drop_rate_stt)

        # ── Fusion head ───────────────────────────────────────────────────────
        # Concatenated [CNN_channels + MLP_channels] → 1 output per timestep
        self.head = nn.Conv1d(hts3 + hstt2, 1, kernel_size=1)

    def forward(self, x_ts, x_static):
        # ── CNN1D branch ──────────────────────────────────────────────────────
        x = x_ts.permute(0, 2, 1)           # [B, T, n_fts] → [B, n_fts, T]
        x = F.pad(x, (6, 0), value=0)
        x = F.relu(self.conv1(x))           # [B, hts1, T]  
        x = self.dropout_ts1(x)
        x = F.pad(x, (4, 0), value=0)
        x = F.relu(self.conv2(x))           # [B, hts2, T]
        x = self.dropout_ts2(x)
        x = F.pad(x, (2, 0), value=0)
        x = F.relu(self.conv3(x))           # [B, hts3, T]
        x = self.dropout_ts3(x)

        # ── MLP branch ────────────────────────────────────────────────────────
        s = F.relu(self.fc1(x_static))      # [B, hstt1]
        s = self.dropout_stt1(s)
        s = F.relu(self.fc2(s))             # [B, hstt2]
        s = self.dropout_stt2(s)

        # ── Broadcast static vector across T timesteps ────────────────────────
        T = x.shape[2]
        s = s.unsqueeze(2).expand(-1, -1, T)    # [B, hstt2] → [B, hstt2, T]

        # ── Fuse and predict ──────────────────────────────────────────────────
        x = torch.cat([x, s], dim=1)        # [B, hts3 + hstt2, T]
        x = self.head(x)                    # [B, 1, T]
        return x.squeeze(1)                 # [B, T]


# =============================================================================
# Two-stage model components (shared backbones + correction heads)
# =============================================================================


class SharedMLP(nn.Module):

    """Shared MLP backbone for the two-stage MLP model.

    Identical to MLP but without the final output layer.  Returns the last
    hidden state so each CorrectionHead can consume it independently.

    Parameters
    ----------
    input_size   : Number of input features.
    h1, h2, hout : Widths of hidden layers 1, 2, and 3.
    drop_rate : Dropout probability applied after each hidden layer.

    Input  : x — [B, input_size]
    Output :   — [B, hout]
    """

    def __init__(self, input_size, h1, h2, hout, drop_rate):    
        super().__init__()
        self.fc1      = nn.Linear(input_size, h1)
        self.dropout1 = nn.Dropout(drop_rate)
        self.fc2      = nn.Linear(h1, h2)
        self.dropout2 = nn.Dropout(drop_rate)
        self.fc3      = nn.Linear(h2, hout)
        self.dropout3 = nn.Dropout(drop_rate)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # [B, h1]
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))  # [B, h2]
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))  # [B, hout]
        x = self.dropout3(x)

        return x                  # [B, hout]


class SharedCNN1D(nn.Module):

    """Shared CNN1D backbone for the two-stage CNN1D model.

    Identical to CNN1D, but without the final 1×1 head layer.  
    Returns per-timestep feature vectors for each CorrectionHead.

    Parameters
    ----------
    in_features  : Number of input channels.
    h1, h2, hout : Channel widths of Conv layers 1, 2, and 3.
    drop_rate    : Dropout probability applied after each conv layer.

    Input  : x — [B, T, in_features]
    Output :   — [B, T, hout]
    """

    def __init__(self, in_features, h1, h2, hout, drop_rate):   
        super().__init__()
        self.conv1    = nn.Conv1d(in_features, h1,   kernel_size=7, padding=0)
        self.dropout1 = nn.Dropout(drop_rate)
        self.conv2    = nn.Conv1d(h1,          h2,   kernel_size=5, padding=0)
        self.dropout2 = nn.Dropout(drop_rate)
        self.conv3    = nn.Conv1d(h2,          hout, kernel_size=3, padding=0)
        self.dropout3 = nn.Dropout(drop_rate)

    def forward(self, x):
        x = x.permute(0, 2, 1)              # [B, T, F] → [B, F, T]
        x = F.pad(x, (6, 0), value=0)       # causal pad → [B, F,  T+6]  
        x = F.relu(self.conv1(x))           # → [B, h1, T]          
        x = self.dropout1(x)
        x = F.pad(x, (4, 0), value=0)       # causal pad → [B, h1, T+4]   
        x = F.relu(self.conv2(x))           # → [B, h2, T]
        x = self.dropout2(x)
        x = F.pad(x, (2, 0), value=0)       # causal pad → [B, h2, T+2] 
        x = F.relu(self.conv3(x))           # → [B, hout, T]             
        x = self.dropout3(x)
        return x.permute(0, 2, 1)           # → [B, T, hout]


class SharedDB(nn.Module):
  
    """Shared DB backbone for the TS-DB model.

    Identical to DB but replaces the scalar output head with
    a multi-channel fusion head, returning a per-timestep feature tensor for
    each CorrectionHead to consume independently.

    Parameters
    ----------
    n_fts            : Number of time-series input channels.
    n_fstatic        : Number of static input features.
    hts1/hts2/hts3  : CNN1D hidden channel widths for layers 1, 2, and 3.
    hstt1/hstt2     : MLP hidden widths for layers 1 and 2.
    hout            : Output channel width after the fusion head.
    drop_rate_ts    : Dropout probability for the CNN1D branch.
    drop_rate_stt   : Dropout probability for the MLP branch.

    Inputs  : x_ts     — [B, T, n_fts]
              x_static — [B, n_fstatic]
    Output  :          — [B, T, hout]
    """

    def __init__(self, n_fts, n_fstatic, hts1, hts2, hts3, hstt1, hstt2, hout, drop_rate_ts, drop_rate_stt):
        super().__init__()

        # ── CNN1D branch (time-series features) ──────────────────────────────
        self.conv1       = nn.Conv1d(n_fts,  hts1, kernel_size=7, padding=0)
        self.dropout_ts1 = nn.Dropout(drop_rate_ts)
        self.conv2       = nn.Conv1d(hts1,  hts2, kernel_size=5, padding=0)
        self.dropout_ts2 = nn.Dropout(drop_rate_ts)
        self.conv3       = nn.Conv1d(hts2,  hts3, kernel_size=3, padding=0)
        self.dropout_ts3 = nn.Dropout(drop_rate_ts)

        # ── MLP branch (static features) ─────────────────────────────────────
        self.fc1          = nn.Linear(n_fstatic, hstt1)
        self.dropout_stt1 = nn.Dropout(drop_rate_stt)
        self.fc2          = nn.Linear(hstt1, hstt2)
        self.dropout_stt2 = nn.Dropout(drop_rate_stt)

        # ── Fusion head ───────────────────────────────────────────────────────
        # Concatenated [CNN_channels + MLP_channels] → hout channels per timestep
        self.head = nn.Conv1d(hts3 + hstt2, hout, kernel_size=1)

    def forward(self, x_ts, x_static):
        # ── CNN1D branch ──────────────────────────────────────────────────────
        x = x_ts.permute(0, 2, 1)           # [B, T, n_fts] → [B, n_fts, T]
        x = F.pad(x, (6, 0), value=0)
        x = F.relu(self.conv1(x))           # [B, hts1, T]    
        x = self.dropout_ts1(x)
        x = F.pad(x, (4, 0), value=0)
        x = F.relu(self.conv2(x))           # [B, hts2, T]
        x = self.dropout_ts2(x)
        x = F.pad(x, (2, 0), value=0)
        x = F.relu(self.conv3(x))           # [B, hts3, T]
        x = self.dropout_ts3(x)

        # ── MLP branch ────────────────────────────────────────────────────────
        s = F.relu(self.fc1(x_static))      # [B, hstt1]
        s = self.dropout_stt1(s)
        s = F.relu(self.fc2(s))             # [B, hstt2]
        s = self.dropout_stt2(s)

        # ── Broadcast static vector across T timesteps ────────────────────────
        T = x.shape[2]
        s = s.unsqueeze(2).expand(-1, -1, T)    # [B, hstt2] → [B, hstt2, T]

        # ── Fuse ──────────────────────────────────────────────────────────────
        x = torch.cat([x, s], dim=1)        # [B, hts3 + hstt2, T]
        x = self.head(x)                    # [B, hout, T]

        # Permute so CorrectionHead can apply Linear layers over the last dimension
        return x.permute(0, 2, 1)           # [B, T, hout]


class CorrectionHead(nn.Module):
    
    """Per-source residual correction head used by all two-stage models.

    Each CMAQ source gets its own CorrectionHead instance.  The head receives:
      - the hidden representation from the stage-1 backbone, and
      - the scalar CMAQ source contribution s_k for that specific source.

    s_k is first projected through a small 2-layer embedding MLP before being
    concatenated with the hidden representation. The combined tensor is then
    passed through two fully-connected layers to produce s_delta (the residual
    correction added to the raw CMAQ source contribution).

    Works for both MLP-style inputs (2-D tensors: [B, hout]) and CNN1D/DB
    inputs (3-D tensors: [B, T, hout]); Linear layers operate over the last
    dimension in both cases.

    Parameters
    ----------
    hout         : Width of the incoming hidded representation.
    hh1, hh2     : Hidden widths of the correction MLP (layers 1 and 2).
    drop_rate    : Dropout probability.
    hsk          : Hidden width of the s_k embedding network (default 32).

    Inputs  : hidded_repr — [B, hout]   or  [B, T, hout]
              s_k         — [B]         or  [B, T]
    Output  :             — [B]        or  [B, T]    (residual s_delta)
    """

    def __init__(self, hout, hh1, hh2, drop_rate, hsk):    
        super().__init__()

        # Small embedding network that projects the scalar CMAQ value s_k
        # into a higher-dimensional space before concatenation
        self.sk_embed = nn.Sequential(
            nn.Linear(1,   hsk),
            nn.ReLU(),
            nn.Linear(hsk, hsk),)    

        self.fc1      = nn.Linear(hout + hsk, hh1)
        self.dropout1 = nn.Dropout(drop_rate)
        self.fc2      = nn.Linear(hh1, hh2)
        self.dropout2 = nn.Dropout(drop_rate)
        self.fc3      = nn.Linear(hh2, 1)

    def forward(self, hidded_repr, s_k):
        # hidded_repr: [B, hout]  or [B, T, hout]
        # s_k:         [B]        or [B, T]
        sk_emb = self.sk_embed(s_k.unsqueeze(-1))       # [B, hsk]  or  [B, T, hsk]
        x = torch.cat([hidded_repr, sk_emb], dim=-1)    # [B, hout+hsk]  or  [B, T, hout+hsk]
        x = F.relu(self.fc1(x))                         # [B, hh1]  or  [B, T, hh1]
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))                         # [B, hh2]  or  [B, T, hh2]
        x = self.dropout2(x)
        x = self.fc3(x).squeeze(-1)                     # [B]  or  [B, T]
        return x


# =============================================================================
# Two-stage models
# =============================================================================


class TwoStageMLP(nn.Module):    
    """Two-stage MLP model.

    Stage 1 — SharedMLP produces a hidden representation from all input features.
    Stage 2 — Each CorrectionHead uses the hidded representation together with its
               corresponding raw CMAQ source contribution to predict s_delta.

    Corrected source contributions (s_cmaq + s_delta) are clamped to a minimum
    threshold to prevent physically implausible negative values, then summed
    to produce the final PM2.5 prediction.

    Parameters
    ----------
    n_sources : Number of CMAQ source contributions (occupies first n_sources
                columns of x).
    n_fkey     : Number of key (CMAQ) input features.
    n_fcontrol : Number of control (meteorological / auxiliary) input features.
    h1/h2/hout: SharedMLP hidden-layer widths.
    dr        : Dropout probability for SharedMLP.
    hh1/hh2   : CorrectionHead hidden-layer widths.
    hdr       : Dropout probability for CorrectionHead.
    hsk       : Hidden width of the s_k embedding network in CorrectionHead.
    threshold : Minimum allowed value for corrected source contributions.

    Input  : x — [B, n_fkey + n_fcontrol]
    Output : (y_pred, s_corrected, s_delta, hidded_repr)
               y_pred      — [B, 1]
               s_corrected — [B, n_sources]
               s_delta     — [B, n_sources]
               hidded_repr — [B, hout]
    """

    def __init__(self, n_sources, n_fkey, n_fcontrol, h1, h2, hout, drop_rate, hh1, hh2, hdrop_rate, hsk, threshold, source_mean=None, source_scale=None, target_mean=None, target_scale=None):
        super().__init__()

        self.n_sources  = n_sources
        self.register_buffer('threshold', torch.tensor(threshold, dtype=torch.float32))
        _optional_tensor_buffer(self, 'source_mean', source_mean, (1, n_sources))  
        _optional_tensor_buffer(self, 'source_scale', source_scale, (1, n_sources))  
        _optional_tensor_buffer(self, 'target_mean', target_mean, (1, 1))  
        _optional_tensor_buffer(self, 'target_scale', target_scale, (1, 1))  
        self.shared_mlp = SharedMLP(n_fkey + n_fcontrol, h1, h2, hout, drop_rate)
        self.heads      = nn.ModuleList(
            [CorrectionHead(hout, hh1, hh2, hdrop_rate, hsk) for _ in range(n_sources)]
        )

    def forward(self, x):
        # Extract raw CMAQ source contributions from the first n_sources columns
        s_cmaq      = x[:, :self.n_sources]    # [B, n_sources]    
        hidded_repr = self.shared_mlp(x)       # [B, hout]         

        # Each head predicts a residual correction (s_delta) for its source
        s_delta = torch.stack(    
            [head(hidded_repr, s_cmaq[:, k]) for k, head in enumerate(self.heads)],
            dim=1,
        )                                       # [B, n_sources]

        s_corrected = s_cmaq + s_delta          # [B, n_sources]

        # Clamp to prevent physically implausible negative source contributions
        s_corrected = torch.clamp(s_corrected, min=self.threshold)  # [B, n_sources]

        # Sum all corrected sources to get the final PM2.5 prediction
        y_pred = _sum_corrected_sources_in_target_scale(s_corrected, self.source_mean, self.source_scale, self.target_mean, self.target_scale)  

        return y_pred, s_corrected, s_delta, hidded_repr


class TwoStageCNN1D(nn.Module):    
    
    """Two-stage CNN1D model.

    Stage 1 — SharedCNN1D produces a hidden representation from all input features.
    Stage 2 — Each CorrectionHead uses the hidded representation together with its
               corresponding raw CMAQ source timeseries to predict s_delta.

    Corrected source contributions are clamped, then summed across sources to
    give a per-timestep PM2.5 prediction.

    Parameters
    ----------
    n_sources : Number of CMAQ source contributions (occupies first n_sources
                channels of x).
    n_fkey     : Number of key (CMAQ) time-series features.
    n_fcontrol : Number of control time-series features.
    h1/h2/hout: SharedCNN1D hidden channel widths.
    dr        : Dropout probability for SharedCNN1D.
    hh1/hh2   : CorrectionHead hidden-layer widths.
    hdr       : Dropout probability for CorrectionHead.
    hsk       : Hidden width of the s_k embedding network in CorrectionHead.
    threshold : Minimum allowed value for corrected source contributions.

    Input  : x — [B, T, n_fkey + n_fcontrol]
    Output : (y_pred, s_corrected, s_delta, hidded_repr)
               y_pred      — [B, T]
               s_corrected — [B, T, n_sources]
               s_delta     — [B, T, n_sources]
               hidded_repr — [B, T, hout]
    """

    def __init__(self, n_sources, n_fkey, n_fcontrol, h1, h2, hout, drop_rate, hh1, hh2, hdrop_rate, hsk, threshold, source_mean=None, source_scale=None, target_mean=None, target_scale=None):
        super().__init__()

        self.n_sources    = n_sources
        self.register_buffer('threshold', torch.tensor(threshold, dtype=torch.float32))
        _optional_tensor_buffer(self, 'source_mean', source_mean, (1, 1, n_sources))  
        _optional_tensor_buffer(self, 'source_scale', source_scale, (1, 1, n_sources)) 
        _optional_tensor_buffer(self, 'target_mean', target_mean, (1, 1))  
        _optional_tensor_buffer(self, 'target_scale', target_scale, (1, 1))  
        self.shared_cnn1d = SharedCNN1D(n_fkey + n_fcontrol, h1, h2, hout, drop_rate)
        self.heads        = nn.ModuleList(
            [CorrectionHead(hout, hh1, hh2, hdrop_rate, hsk) for _ in range(n_sources)]
        )

    def forward(self, x):
        # Extract raw CMAQ source contributions from the first n_sources channels
        s_cmaq      = x[:, :, :self.n_sources]    # [B, T, n_sources]    
        hidded_repr = self.shared_cnn1d(x)        # [B, T, hout]         

        # Each head predicts a residual correction (s_delta) for its source
        s_delta = torch.stack(   
            [head(hidded_repr, s_cmaq[:, :, k]) for k, head in enumerate(self.heads)],
            dim=-1,
        )                                          # [B, T, n_sources]

        s_corrected = s_cmaq + s_delta             # [B, T, n_sources]

        # Clamp to prevent physically implausible negative source contributions
        s_corrected = torch.clamp(s_corrected, min=self.threshold)  # [B, T, n_sources]    

        # Sum all corrected sources to get the final per-timestep PM2.5 prediction
        y_pred = _sum_corrected_sources_in_target_scale(s_corrected, self.source_mean, self.source_scale, self.target_mean, self.target_scale) 

        return y_pred, s_corrected, s_delta, hidded_repr


class TwoStageDB(nn.Module):    
    
    """Two-stage DB model.

    Stage 1 — SharedDB produces per-timestep a hidden representation from
               time-series (CMAQ/WRF) and static (land-use) features.
    Stage 2 — Each CorrectionHead uses the hidded representation together with its
               corresponding raw CMAQ source timeseries to predict s_delta.

    Corrected source contributions are clamped, then summed across sources to
    give a per-timestep PM2.5 prediction.

    Parameters
    ----------
    n_sources       : Number of CMAQ source contributions (first n_sources
                      channels of x_ts).
    n_fts            : Total number of time-series input channels.
    n_fstatic        : Number of static input features.
    hts1/hts2/hts3  : CNN1D hidden channel widths (layers 1-3) in SharedDB.
    hstt1/hstt2     : MLP hidden widths (layers 1-2) in SharedDB.
    drop_rate_ts    : Dropout probability for the CNN1D branch.
    drop_rate_stt   : Dropout probability for the MLP branch.
    hh1/hh2         : CorrectionHead hidden-layer widths.
    hout            : SharedDB fusion output width; also the CorrectionHead
                      input width.
    hdrop_rate      : Dropout probability for CorrectionHead.
    threshold       : Minimum allowed value for corrected source contributions.
    hsk             : Hidden width of the s_k embedding network in CorrectionHead.

    Inputs  : x_ts     — [B, T, n_fts]
              x_static — [B, n_fstatic]
    Output  : (y_pred, s_corrected, s_delta, hidded_repr)
               y_pred      — [B, T]
               s_corrected — [B, T, n_sources]
               s_delta     — [B, T, n_sources]
               hidded_repr — [B, T, hout]
    """

    def __init__(
        self,
        n_sources,
        n_fts, n_fstatic,
        hts1, hts2, hts3,
        hstt1, hstt2,
        drop_rate_ts, drop_rate_stt,
        hh1, hh2, hout,
        hdrop_rate, hsk,
        threshold,
        source_mean=None, source_scale=None, target_mean=None, target_scale=None,
    ):
        super().__init__()

        self.n_sources = n_sources
        self.register_buffer('threshold', torch.tensor(threshold, dtype=torch.float32))
        _optional_tensor_buffer(self, 'source_mean', source_mean, (1, 1, n_sources))  
        _optional_tensor_buffer(self, 'source_scale', source_scale, (1, 1, n_sources))  
        _optional_tensor_buffer(self, 'target_mean', target_mean, (1, 1))  
        _optional_tensor_buffer(self, 'target_scale', target_scale, (1, 1))  
        self.shared_db = SharedDB(
            n_fts, n_fstatic, hts1, hts2, hts3, hstt1, hstt2, hout, drop_rate_ts, drop_rate_stt
        )
        self.heads = nn.ModuleList(
            [CorrectionHead(hout, hh1, hh2, hdrop_rate, hsk) for _ in range(n_sources)]
        )

    def forward(self, x_ts, x_static):
        # x_ts:     [B, T, F_ts]    — first n_sources channels are CMAQ source contributions
        # x_static: [B, F_static]

        # Extract raw CMAQ source contributions from the first n_sources channels
        s_cmaq      = x_ts[:, :, :self.n_sources]              # [B, T, n_sources]
        hidded_repr = self.shared_db(x_ts, x_static)           # [B, T, hout]

        # Each head predicts a residual correction (s_delta) for its source
        s_delta = torch.stack(   
            [head(hidded_repr, s_cmaq[:, :, k]) for k, head in enumerate(self.heads)],
            dim=-1,
        )                                                       # [B, T, n_sources]

        s_corrected = s_cmaq + s_delta                          # [B, T, n_sources]

        # Clamp to prevent physically implausible negative source contributions
        s_corrected = torch.clamp(s_corrected, min=self.threshold)  # [B, T, n_sources]    

        # Sum all corrected sources to get the final per-timestep PM2.5 prediction
        y_pred = _sum_corrected_sources_in_target_scale(s_corrected, self.source_mean, self.source_scale, self.target_mean, self.target_scale) 

        return y_pred, s_corrected, s_delta, hidded_repr
