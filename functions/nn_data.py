"""Data preparation and prediction collection utilities for the PM2.5 source-attribution pipeline.

Public API
----------
prepare_X_y                      : Extract feature matrix and target vector from a DataFrame.
MLPDataset                       : PyTorch Dataset for tabular (MLP) training data.
prepare_3d_input                 : Build per-station 3-D arrays grouped by sequence length.
inverse_sources                  : Inverse-transform CMAQ source columns from standardised space.
collect_CNN1D_preds              : Inference + prediction DataFrame for CNN1D.
collect_DB_preds                 : Inference + prediction DataFrame for DB.
collect_TwoStageMLP_preds        : Inference + prediction DataFrame for TwoStageMLP.
collect_TwoStageCNN1D_preds      : Inference + prediction DataFrame for TwoStageCNN1D.
collect_TwoStageDB_preds         : Inference + prediction DataFrame for TwoStageDB.

Private helpers (prefix '_') are shared across the collect_* functions and are not
intended to be called directly.
"""

import torch
from torch.utils.data import Dataset
import math
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
from fill_missing_val import *

DATA_DIR  = 'data_path'
ST_DIR    = 'data_path'

CMAQ_BL_FEATURES = ['cmaq_pm25']
CMAQ_FEATURES    = ['cmaq_pm25_bb', 'cmaq_pm25_anthro', 'cmaq_pm25_othr']
AOD_FEATURES     = ['aod']
WRF_FEATURES     = ['wrf_pbl', 'wrf_temp', 'wrf_wspd', 'wrf_rh']
LU_FEATURES      = ['population', 'elevation', 'builtup', 
                    'rice', 'forest', 'minor_road_length', 'major_road_length']

KEY_FEATURES     = CMAQ_FEATURES + AOD_FEATURES 
CONTROL_FEATURES = WRF_FEATURES + LU_FEATURES 

FEATURES         = KEY_FEATURES + CONTROL_FEATURES     
FEATURES_BL      = CMAQ_BL_FEATURES + CONTROL_FEATURES
FEATURES_AB      = CMAQ_FEATURES + CONTROL_FEATURES 
TS_FEATURES      = CMAQ_FEATURES + AOD_FEATURES + WRF_FEATURES 
STATIC_FEATURES  = LU_FEATURES

N_SOURCES  = len(CMAQ_FEATURES)
N_FKEY     = len(KEY_FEATURES)
N_FCONTROL = len(CONTROL_FEATURES)
N_FTS      = len(TS_FEATURES)
N_FSTATIC  = len(STATIC_FEATURES)

# =============================================================================
# Tabular (MLP) utilities
# =============================================================================
def stratified_random_sampling(df, test_size, seed=51, min_per_side=1):
    """
    Station-level stratified random sampling by region using the largest-
    remainder (Hare) apportionment method. 
    Regions with only 1 station are forced entirely into train.
    
    """
    region_sizes           = df['region5'].value_counts()
    forced_train_regions   = region_sizes[region_sizes == 1].index.tolist()
    eligible_regions       = region_sizes[region_sizes > 1]

    n_eligible    = eligible_regions.sum()
    # Target is computed over splittable stations only, so forced-train
    #     regions don't silently shrink your achievable test ratio.
    n_test_target = round(n_eligible * test_size)

    # Step 1 — exact proportional quota per region
    quotas = (eligible_regions / n_eligible) * n_test_target

    # Step 2 — floor each quota, clamp to [min_per_side, n_i - min_per_side]
    alloc = {}
    for region, n_i in eligible_regions.items():
        n_i_test = max(int(np.floor(quotas[region])), min_per_side)
        n_i_test = min(n_i_test, n_i - min_per_side)
        alloc[region] = n_i_test

    # Step 3 — hand out remaining seats to largest fractional remainders
    remainders = {r: quotas[r] - math.floor(quotas[r]) for r in eligible_regions.index}
    deficit    = n_test_target - sum(alloc.values())

    for region in sorted(remainders, key=remainders.get, reverse=True):
        if deficit <= 0:
            break
        cap = eligible_regions[region] - min_per_side
        if alloc[region] < cap:
            alloc[region] += 1
            deficit -= 1

    # Step 4 — perform the actual random draw within each region
    df_train_all = [df[df['region5'] == r] for r in forced_train_regions]
    df_test_all  = []

    for region, n_i_test in alloc.items():
        sub_df = df[df['region5'] == region]
        df_tr, df_te = train_test_split(sub_df, test_size=n_i_test, random_state=seed)
        df_train_all.append(df_tr)
        df_test_all.append(df_te)

    train_stations = pd.concat(df_train_all, ignore_index=True)
    test_stations  = pd.concat(df_test_all,  ignore_index=True)

    achieved = len(test_stations) / (len(train_stations) + len(test_stations))
    print(f"Target test ratio   : {test_size:.3f}")
    print(f"Achieved (stations) : {achieved:.3f}  "
          f"({len(test_stations)} test / {len(train_stations)+len(test_stations)} total)")

    return train_stations, test_stations

def load_pm25_data(pm25_dir=DATA_DIR, features=FEATURES, split_val=False,
                   val_size=0.2, test_size=0.2, seed=51):
    """
    Load, split, and impute PM2.5 data.

    split_val=False (default):  2-way split -- returns df_train, df_test.
    split_val=True:             3-way split -- returns df_train, df_val, df_test.
    """
    df         = pd.read_csv(pm25_dir)
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df['year'] = df['date'].dt.year
    df = df[features + ['region5', 'stcode', 'date', 'pm25', 'lat', 'lon']]
    st = df[['stcode', 'region5']].drop_duplicates().reset_index(drop=True)
    train_stations, test_stations = stratified_random_sampling(st, test_size=test_size, seed=seed)
    df_test = df[df['stcode'].isin(test_stations['stcode'])].copy()

    if split_val:
        st_train = st[st['stcode'].isin(train_stations['stcode'])]
        train_stations, val_stations = stratified_random_sampling(st_train, test_size=val_size, seed=seed)
        df_val   = df[df['stcode'].isin(val_stations['stcode'])].copy()
        df_train = df[df['stcode'].isin(train_stations['stcode'])].copy()
        df_train = fill_missing_values(df_train)

        return df_train, df_val, df_test

    df_train = df[df['stcode'].isin(train_stations['stcode'])].copy()
    df_train = fill_missing_values(df_train)

    return df_train, df_test


def load_pm25_data_region_holdout(pm25_dir=DATA_DIR, features=FEATURES, 
                                 region='Central'):
    """
    Load and split PM2.5 data into train/test using region hold-out:
    every station located in `region` goes entirely to the test set; every
    other station goes entirely to train.

    region : str or list of str
        Region name(s) (must match values in 'region5') to hold out for test.

    Returns df_train, df_test.
    """
    df    = pd.read_csv(pm25_dir)
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df['year'] = df['date'].dt.year
    df = df[features + ['region5', 'stcode', 'date', 'pm25', 'lat', 'lon']]

    held_out_regions = [region] if isinstance(region, str) else list(region)

    available_regions = set(df['region5'].dropna().unique())
    unknown = set(held_out_regions) - available_regions
    if unknown:
        raise ValueError(f"Unknown region(s) {unknown}; available regions are {sorted(available_regions)}")

    test_mask = df['region5'].isin(held_out_regions)
    df_test  = df[test_mask]
    df_train = df[~test_mask]

    df_train = fill_missing_values(df_train)

    print(f"Held-out region(s)  : {held_out_regions}")
    print(f"Train stations      : {df_train['stcode'].nunique()}  ({len(df_train):,} rows)")
    print(f"Test stations       : {df_test['stcode'].nunique()}  ({len(df_test):,} rows)")

    return df_train, df_test

class MLPDataset(Dataset):
    
    """PyTorch Dataset for tabular (MLP) training data.

    Applies feature and target standardisation at construction time so every
    DataLoader batch automatically receives scaled tensors.

    Parameters
    ----------
    X              : np.ndarray [N, F]  — raw feature matrix.
    y              : np.ndarray [N]     — raw target values.
    feature_scaler : fitted scaler (e.g. StandardScaler) for X.
    target_scaler  : fitted scaler (e.g. StandardScaler) for y.

    __getitem__ returns (x_tensor [F], y_tensor [1]).
    """

    def __init__(self, X, y, feature_scaler, target_scaler):
        X = feature_scaler.transform(X)
        y = target_scaler.transform(y.reshape(-1, 1)).ravel()

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =============================================================================
# Sequence data preparation
# =============================================================================

def prepare_3d_input(    
    df,
    ts_cols,
    static_cols=None,
    target_col='pm25',
    group_col='stcode',
    date_col='date',
    verbose=True,
):
    """Prepare 3-D input arrays for a sequence model or db model.

    Stations with different numbers of observations land in separate length
    groups so they can be batched without padding.

    Parameters
    ----------
    df          : pandas.DataFrame with time-series and optional static columns.
    ts_cols     : list of str — column names for time-series features.
    static_cols : list of str or None.
                  None  → returns CNN1D-style arrays  {'X', 'y', 'stations'}.
                  list  → returns db arrays   {'X_ts', 'X_static', 'y', 'stations'}.
    target_col  : str — target column (default 'pm25_filled').
    group_col   : str — column that identifies sequences, e.g. station code (default 'stcode').
    date_col    : str — column used for chronological ordering (default 'date').
    verbose     : bool — if True, prints shape info for each length group.

    Returns
    -------
    length_groups : dict
        Keys   : int — sequence length T.
        Values : dict with numpy arrays.
            CNN1D mode      → {'X': [N, T, F],       'y': [N, T], 'stations': list}
            db mode → {'X_ts': [N, T, F_ts], 'X_static': [N, F_static],
                               'y': [N, T], 'stations': list}
    """
    
    # Ensure chronological order within each station before grouping.
    df = df.sort_values([group_col, date_col]).reset_index(drop=True)

    is_db = static_cols is not None

    # create_group_dict is a closure that reads 'is_db' from the enclosing
    # scope. defaultdict calls it to initialise the dict for each new sequence length.
    def create_group_dict():
        if is_db:
            return {'X_ts': [], 'X_static': [], 'y': [], 'stations': []}
        return {'X': [], 'y': [], 'stations': []}

    length_groups = defaultdict(create_group_dict)

    # 1. Group data by station and accumulate arrays into lists.
    for stcode, station_df in df.groupby(group_col):
        T = len(station_df)

        length_groups[T]['y'].append(station_df[target_col].values)
        length_groups[T]['stations'].append(stcode)

        if is_db:
            length_groups[T]['X_ts'].append(station_df[ts_cols].values)
            # Static features are time-invariant; only the first row is needed.
            length_groups[T]['X_static'].append(station_df[static_cols].values[0])
        else:
            length_groups[T]['X'].append(station_df[ts_cols].values)

    # 2. Stack the per-station lists into numpy arrays.
    for T, group in length_groups.items():
        group['y'] = np.stack(group['y'], axis=0)  # [N, T]

        if is_db:
            group['X_ts']     = np.stack(group['X_ts'],     axis=0)  # [N, T, F_ts]
            group['X_static'] = np.stack(group['X_static'], axis=0)  # [N, F_static]

            if verbose:
                print(f"Length {T:4d} days → {group['X_ts'].shape[0]:3d} sequences | "
                      f"X_ts: {group['X_ts'].shape} | X_static: {group['X_static'].shape} | "
                      f"y: {group['y'].shape}")
        else:
            group['X'] = np.stack(group['X'], axis=0)  # [N, T, F]

            if verbose:
                print(f"Length {T:4d} days → {group['X'].shape[0]:3d} sequences | "
                      f"X: {group['X'].shape} | y: {group['y'].shape}")

    return length_groups

def source_sum_scaler_params(source_scaler, target_scaler, n_sources=3):  
    """Return scaler statistics needed by two-stage models to sum corrected sources in raw units. 

    The two-stage models receive standardized source inputs, but the corrected
    sources should be summed in raw µg/m³ and then converted back to the
    standardized PM2.5 target scale for loss calculation.
    """
    return { 
        'source_mean':  source_scaler.mean_[:n_sources],   
        'source_scale': source_scaler.scale_[:n_sources],  
        'target_mean':  target_scaler.mean_[0],            
        'target_scale': target_scaler.scale_[0],           
    }

def transform_target_preserve_nan(series, scaler):  
    """Standardise a target Series while preserving missing raw observations. 

    Validation/test sequence models keep rows with missing raw pm25 so temporal
    spacing is not compressed. StandardScaler.transform is applied only to
    observed targets; missing targets remain NaN and are ignored later by the
    masked validation loss and evaluation metrics.
    """
    arr = series.to_numpy(dtype=float).reshape(-1, 1)  
    out = np.full(arr.shape, np.nan, dtype=float) 
    mask = np.isfinite(arr[:, 0])  
    out[mask] = scaler.transform(arr[mask])  
    return out.ravel()  

def groups_to_tensors(groups, mode='db', reverse_sort=True):
    """Convert length-grouped numpy arrays from prepare_3d_input into tensors.

    Parameters
    ----------
    groups       : dict — output of prepare_3d_input (already scaled).
    mode         : 'db' or 'cnn1d'.
    reverse_sort : bool — if True, sort groups from longest to shortest
                   sequence (useful for training; default True).

    Returns
    -------
    dict mapping sequence length T to:
        db → (X_ts [N,T,F_ts], X_static [N,F_static], y [N,T])
        cnn1d      → (X [N,T,F], y [N,T])
    """
    tensors = {}
    for T, group in groups.items():
        if mode == 'db':
            tensors[T] = (
                torch.tensor(group['X_ts'],     dtype=torch.float32),  # [N, T, F_ts]
                torch.tensor(group['X_static'], dtype=torch.float32),  # [N, F_static]
                torch.tensor(group['y'],        dtype=torch.float32),  # [N, T]
            )
        elif mode == 'cnn1d':
            tensors[T] = (
                torch.tensor(group['X'], dtype=torch.float32),         # [N, T, F]
                torch.tensor(group['y'], dtype=torch.float32),         # [N, T]
            )
        else:
            raise ValueError(f"Unknown mode '{mode}'. Expected 'db' or 'cnn1d'.")

    if reverse_sort:
        tensors = dict(sorted(tensors.items(), key=lambda x: x[0], reverse=True))
    return tensors


# =============================================================================
# Inverse-transform utilities
# =============================================================================

def inverse_sources(arr_std, scaler, n_total_features):
    """Inverse-transform CMAQ source columns from standardised space to µg/m³.

    A StandardScaler fitted on the full feature matrix expects an input of width
    n_total_features.  This helper reconstructs that dummy matrix, inserts the K
    source columns at positions 0..K-1, applies inverse_transform, then extracts
    the relevant columns.

    Parameters
    ----------
    arr_std          : np.ndarray [N, K] or [N, T, K] — standardised source values.
    scaler           : fitted scaler with an inverse_transform method.
    n_total_features : int — total width the scaler was fitted on.

    Returns
    -------
    np.ndarray — same shape as arr_std, values in original units (µg/m³).
    """
    
    is_3d = arr_std.ndim == 3
    if is_3d:
        N, T, K = arr_std.shape
        arr_std = arr_std.reshape(N * T, K)         # [N*T, K]

    K = arr_std.shape[1]
    dummy = np.zeros((len(arr_std), n_total_features))  # match scaler's expected width
    dummy[:, :K] = arr_std
    result = scaler.inverse_transform(dummy)[:, :K]

    if is_3d:
        result = result.reshape(N, T, K)            # [N, T, K]
    return result


def _inverse_pred_sequence(pred_np, target_scaler, T):
    """Inverse-transform and reshape a block of standardised sequence predictions.

    Flattens [N, T] → [N*T, 1] for the scaler, then reshapes the result back to
    [N, T].

    Parameters
    ----------
    pred_np       : np.ndarray [N, T] — standardised predictions.
    target_scaler : fitted scaler with an inverse_transform method.
    T             : int — sequence length used to reshape the output.

    Returns
    -------
    np.ndarray [N, T] in original units.
    """

    return target_scaler.inverse_transform(pred_np.reshape(-1, 1)).reshape(-1, T)


# =============================================================================
# Private record-building helpers (shared across collect_* functions)
# =============================================================================

def _get_station_dates(df_orig, stcode):
    """Return chronologically sorted observation dates for a single station.

    Parameters
    ----------
    df_orig : pandas.DataFrame with at minimum 'stcode' and 'date' columns.
    stcode  : station identifier to filter on.

    Returns
    -------
    np.ndarray of date values sorted in ascending order.
    """

    return (df_orig[df_orig['stcode'] == stcode]
            .sort_values('date')['date'].values)


def _build_sequence_records(pred_inv, stations, df_orig, T):
    """Build a flat list of {stcode, date, pred} records for one length group.

    Used by collect_CNN1D_preds and collect_DB_preds.

    Parameters
    ----------
    pred_inv : np.ndarray [N, T] — predictions in original units.
    stations : list of station identifiers (length N).
    df_orig  : pandas.DataFrame with 'stcode' and 'date' columns.
    T        : int — sequence length.

    Returns
    -------
    list of dicts, each with keys 'stcode', 'date', 'pred'.
    """
    
    records = []
    for i, stcode in enumerate(stations):
        dates = _get_station_dates(df_orig, stcode)
        for t in range(T):
            records.append({
                'stcode': stcode,
                'date':   dates[t],
                'pred':   pred_inv[i, t],
            })
    return records


def _build_twostage_sequence_records(
    pred_inv, s_corrected_inv, s_delta_inv, stations, source_names, df_orig, T
):
    """Build per-timestep records that include source correction columns.

    Used by collect_TwoStageCNN1D_preds and collect_TwoStageDB_preds.

    Parameters
    ----------
    pred_inv        : np.ndarray [N, T]    — total PM2.5 predictions in original units.
    s_corrected_inv : np.ndarray [N, T, K] — corrected source contributions in µg/m³.
    s_delta_inv     : np.ndarray [N, T, K] — source residual corrections in µg/m³.
    stations        : list of station identifiers (length N).
    source_names    : list of str (length K) — name of each CMAQ source.
    df_orig         : pandas.DataFrame with 'stcode' and 'date' columns.
    T               : int — sequence length.

    Returns
    -------
    list of dicts with keys 'stcode', 'date', 'pred',
    plus 'corrected_<src>' and 'delta_<src>' for every entry in source_names.
    """

    records = []
    for i, stcode in enumerate(stations):
        dates = _get_station_dates(df_orig, stcode)
        for t in range(T):
            record = {
                'stcode': stcode,
                'date':   dates[t],
                'pred':   pred_inv[i, t],
            }
            for k, src in enumerate(source_names):
                record[f'corrected_{src}'] = s_corrected_inv[i, t, k]
                record[f'delta_{src}']     = s_delta_inv[i, t, k]
            records.append(record)
    return records


def _records_to_pred_df(records, df_orig, target_col='pm25', extra_merge_cols=None):
    """Convert accumulated prediction records into a merged DataFrame.

    Merges the records with ground-truth (and optionally raw source) columns
    from df_orig so the caller receives a single aligned output DataFrame.

    Parameters
    ----------
    records          : list of dicts — produced by a _build_*_records helper.
    df_orig          : pandas.DataFrame — source of ground-truth columns.
    extra_merge_cols : list of str or None.
                       Additional columns to pull from df_orig beyond
                       ['stcode', 'date', 'pm25_filled'], e.g. raw CMAQ source
                       columns for two-stage models.  Pass None (default) for
                       single-stage models.

    Returns
    -------
    pandas.DataFrame with all prediction records left-merged with df_orig.
    """
    
    merge_cols = ['stcode', 'date', target_col]
    if extra_merge_cols:
        merge_cols = merge_cols + list(extra_merge_cols)   
    df_pred = pd.DataFrame(records)
    return df_pred.merge(df_orig[merge_cols], on=['stcode', 'date'], how='left')


# =============================================================================
# Prediction collectors — one per model type
# =============================================================================
def collect_MLP_preds(X_tensor, df_orig, model, target_scaler, target_col='pm25'):
    """Run a trained MLP model and return a prediction DataFrame.

    Parameters
    ----------
    X_tensor      : torch.Tensor [N, F] — scaled input features.
    df_orig       : pandas.DataFrame — original data; provides dates and ground truth.
    model         : trained MLP model.
    target_scaler : fitted scaler used to standardise the training target.

    Returns
    -------
    pandas.DataFrame with columns: date, stcode, pm25, pred.
    """
    model.eval()
    with torch.no_grad():
        pred     = model(X_tensor)                                    # [N, 1]
        pred_inv = target_scaler.inverse_transform(pred.numpy())      # [N, 1]

    df_pred = (
        df_orig[['date', 'stcode', target_col]]
        .assign(pred=pred_inv.squeeze())
    )
    return df_pred
    
def collect_CNN1D_preds(groups, df_orig, model, target_scaler, target_col='pm25'):
    """Run a trained CNN1D model on all length groups and return a prediction DataFrame.

    Parameters
    ----------
    groups        : dict — output of prepare_3d_input (CNN1D mode).
    df_orig       : pandas.DataFrame — original data; provides dates and ground truth.
    model         : trained CNN1D model.
    target_scaler : fitted scaler used to standardise the training target.

    Returns
    -------
    pandas.DataFrame with columns: stcode, date, pred, pm25_filled.
    """
    
    records = []
    model.eval()
    with torch.no_grad():
        for T, group in groups.items():
            X_tensor = torch.tensor(group['X'], dtype=torch.float32)          # [N, T, F]
            pred     = model(X_tensor).numpy()                                # [N, T]
            pred_inv = _inverse_pred_sequence(pred, target_scaler, T)         # [N, T]    
            records.extend(                                                   
                _build_sequence_records(pred_inv, group['stations'], df_orig, T)
            )
    return _records_to_pred_df(records, df_orig, target_col)                              


def collect_DB_preds(groups, df_orig, model, target_scaler, target_col='pm25'):
    """Run a trained DB model on all length groups and return a prediction DataFrame.

    Parameters
    ----------
    groups        : dict — output of prepare_3d_input (db mode).
    df_orig       : pandas.DataFrame — original data.
    model         : trained DB model.
    target_scaler : fitted scaler for the target.

    Returns
    -------
    pandas.DataFrame with columns: stcode, date, pred, pm25_filled.
    """
    
    records = []
    model.eval()
    with torch.no_grad():
        for T, group in groups.items():
            X_ts_tensor     = torch.tensor(group['X_ts'],     dtype=torch.float32)  # [N, T, F_ts]
            X_static_tensor = torch.tensor(group['X_static'], dtype=torch.float32)  # [N, F_static]
            pred     = model(X_ts_tensor, X_static_tensor).numpy()                  # [N, T]
            pred_inv = _inverse_pred_sequence(pred, target_scaler, T)               # [N, T]    
            records.extend(                                                         
                _build_sequence_records(pred_inv, group['stations'], df_orig, T)
            )
    return _records_to_pred_df(records, df_orig, target_col)                                 


def collect_TwoStageMLP_preds(    
    X_tensor,
    df_orig,
    model,
    target_scaler,
    feature_scaler,
    source_names,
    n_total_features,
    n_sources=3,  
    target_col='pm25',
):
    """Run a trained TwoStageMLP model and return predictions with source corrections.

    Parameters
    ----------
    X_tensor         : torch.Tensor [N, F] — scaled input features.
    df_orig          : pandas.DataFrame — original data with ground truth and raw source columns.
    model            : trained TwoStageMLP model.
    target_scaler    : fitted scaler for the PM2.5 target.
    feature_scaler   : fitted scaler for the full feature matrix (used to invert sources).
    source_names     : list of str (length K) — CMAQ source column names.
    n_total_features : int — total number of features the scaler was fitted on.
    n_sources        : int — CMAQ source columns occupy X[:, :n_sources] (default 3).

    Returns
    -------
    pandas.DataFrame with columns:
        date, stcode, pm25_filled, <source_names>, pred,
        corrected_<src> and delta_<src> for each source in source_names.
    """

    model.eval()
    with torch.no_grad():
        pred, s_corrected, s_delta, _ = model(X_tensor)
        pred_inv        = target_scaler.inverse_transform(pred.numpy())                           # [N, 1]
        s_corrected_inv = inverse_sources(s_corrected.numpy(), feature_scaler, n_total_features)  # [N, K]
        s_cmaq_std      = X_tensor[:, :n_sources].numpy()                                         # [N, K]    
        s_cmaq_orig     = inverse_sources(s_cmaq_std, feature_scaler, n_total_features)           # [N, K]
        s_delta_inv     = s_corrected_inv - s_cmaq_orig                                           # [N, K] µg/m³

    df_pred = (
        df_orig[['date', 'stcode', target_col] + source_names]   
        .assign(pred=pred_inv.squeeze())
    )
    for i, src in enumerate(source_names):
        df_pred[f'corrected_{src}'] = s_corrected_inv[:, i]
        df_pred[f'delta_{src}']     = s_delta_inv[:, i]

    return df_pred

def collect_TwoStageCNN1D_preds(    
    groups,
    df_orig,
    model,
    target_scaler,
    feature_scaler,
    source_names,
    n_total_features,
    n_sources=3,   
    target_col='pm25'
):
    """Run a trained TwoStageCNN1D model on all length groups and return predictions.

    Parameters
    ----------
    groups           : dict — output of prepare_3d_input (CNN1D mode).
    df_orig          : pandas.DataFrame — original data.
    model            : trained TwoStageCNN1D model.
    target_scaler    : fitted scaler for the PM2.5 target.
    feature_scaler   : fitted scaler for the full time-series feature matrix.
    source_names     : list of str (length K) — CMAQ source column names.
    n_total_features : int — total number of features the scaler was fitted on.
    n_sources        : int — CMAQ source channels occupy X[:, :, :n_sources] (default 3).

    Returns
    -------
    pandas.DataFrame with columns:
        stcode, date, pm25_filled, <source_names>,
        pred, corrected_<src> and delta_<src> for each source in source_names.
    """
    
    records = []
    model.eval()
    with torch.no_grad():
        for T, group in groups.items():
            X_tensor        = torch.tensor(group['X'], dtype=torch.float32)                           # [N, T, F]
            pred, s_corrected, s_delta, _ = model(X_tensor)
            pred_inv        = _inverse_pred_sequence(pred.numpy(), target_scaler, T)                  # [N, T]   
            s_corrected_inv = inverse_sources(s_corrected.numpy(), feature_scaler, n_total_features)  # [N, T, K]
            s_cmaq_std      = X_tensor[:, :, :n_sources].numpy()                                      # [N, T, K]    
            s_cmaq_orig     = inverse_sources(s_cmaq_std, feature_scaler, n_total_features)           # [N, T, K]
            s_delta_inv     = s_corrected_inv - s_cmaq_orig                                           # [N, T, K] µg/m³    
            records.extend(                                                                            
                _build_twostage_sequence_records(
                    pred_inv, s_corrected_inv, s_delta_inv,
                    group['stations'], source_names, df_orig, T,
                )
            )
    return _records_to_pred_df(records, df_orig, target_col, extra_merge_cols=source_names)    


def collect_TwoStageDB_preds(   
    groups,
    df_orig,
    model,
    target_scaler,
    ts_scaler,
    source_names,
    n_total_features,
    n_sources=3,   
    target_col='pm25',
):
    """Run a trained TwoStageDB model on all length groups and return predictions.

    Parameters
    ----------
    groups           : dict — output of prepare_3d_input (db mode).
    df_orig          : pandas.DataFrame — original data.
    model            : trained TwoStageDB model.
    target_scaler    : fitted scaler for the PM2.5 target.
    ts_scaler        : fitted scaler for the time-series feature matrix
                       (used to inverse-transform CMAQ source columns).
    source_names     : list of str (length K) — CMAQ source column names.
    n_total_features : int — total number of time-series features the scaler was fitted on.
    n_sources        : int — CMAQ source channels occupy X_ts[:, :, :n_sources] (default 3).

    Returns
    -------
    pandas.DataFrame with columns:
        stcode, date, pm25_filled, <source_names>,
        pred, corrected_<src> and delta_<src> for each source in source_names.
    """
    
    records = []
    model.eval()
    with torch.no_grad():
        for T, group in groups.items():
            X_ts_tensor     = torch.tensor(group['X_ts'],     dtype=torch.float32)                # [N, T, F_ts]
            X_static_tensor = torch.tensor(group['X_static'], dtype=torch.float32)                # [N, F_static]
            pred, s_corrected, s_delta, _ = model(X_ts_tensor, X_static_tensor)
            pred_inv        = _inverse_pred_sequence(pred.numpy(), target_scaler, T)               # [N, T]    
            s_corrected_inv = inverse_sources(s_corrected.numpy(), ts_scaler, n_total_features)   # [N, T, K]
            s_cmaq_std      = X_ts_tensor[:, :, :n_sources].numpy()                               # [N, T, K]   
            s_cmaq_orig     = inverse_sources(s_cmaq_std, ts_scaler, n_total_features)            # [N, T, K]
            s_delta_inv     = s_corrected_inv - s_cmaq_orig                                       # [N, T, K] µg/m³    
            records.extend(                                                                        
                _build_twostage_sequence_records(
                    pred_inv, s_corrected_inv, s_delta_inv,
                    group['stations'], source_names, df_orig, T,
                )
            )
    return _records_to_pred_df(records, df_orig, target_col, extra_merge_cols=source_names)   
