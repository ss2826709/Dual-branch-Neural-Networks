import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from tqdm import tqdm

def convert_distance(lat1, lon1, lat2, lon2):
    """Compute Haversine distance (km) between two lat/lon points."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    R = 6371.0
    return R * c


def build_distance_lookup(df_coord, n_neighbors=5):
    """
    Precompute the nearest N neighbors for every station.

    Returns a dict: {stcode -> [(distance, stcode), ...]} sorted ascending.
    """
    all_stations = df_coord['stcode'].unique().tolist()
    lookup = {}

    for target_station in tqdm(all_stations, desc="Building distance lookup..."):
        target_row = df_coord[df_coord['stcode'] == target_station]
        if target_row.empty:
            lookup[target_station] = []
            continue

        lat1, lon1 = target_row['lat'].values[0], target_row['lon'].values[0]
        distances = []

        for candidate_station in all_stations:
            if candidate_station == target_station:
                continue
            candidate_row = df_coord[df_coord['stcode'] == candidate_station]
            if candidate_row.empty:
                continue
            lat2, lon2 = candidate_row['lat'].values[0], candidate_row['lon'].values[0]
            distance = convert_distance(lat1, lon1, lat2, lon2)
            distances.append((distance, candidate_station))

        lookup[target_station] = sorted(distances)[:n_neighbors]

    return lookup


def fill_missing_val(df_pm25, distance_lookup, target_station, target_date, 
                     return_count=False):
    
    """
    Inverse distance weighted average using up to n_neighbors VALID
    (non-missing) neighbor stations, closest first.
    """
    
    nearest_neighbors = distance_lookup.get(target_station, [])

    weighted_pm_sum   = 0.0
    total_weight      = 0.0
    n_valid_neighbors = 0  

    for distance, station in nearest_neighbors:
        if distance == 0:
            continue  # avoid division by zero for overlapping coordinates

        weight = 1.0 / distance

        neighbor_on_date = (df_pm25['stcode'] == station) & (df_pm25['date'] == target_date)
        neighbor_df = df_pm25[neighbor_on_date]

        if not neighbor_df.empty:
            pm25_value = neighbor_df['pm25'].values[0]
            if not np.isnan(pm25_value):
                weighted_pm_sum += pm25_value * weight
                total_weight += weight
                n_valid_neighbors += 1  

    idw_value = weighted_pm_sum / total_weight if total_weight > 0 else np.nan
 
    if return_count:
        return idw_value, n_valid_neighbors

    return idw_value

def fill_missing_values(df):
    df_coord = df[['stcode', 'lat', 'lon']].drop_duplicates('stcode').reset_index(drop=True)
    # Precompute nearest neighbors for all stations before the main loop
    distance_lookup = build_distance_lookup(df_coord, n_neighbors=5)
    # Identify all rows with missing PM2.5 values
    df_missing = df[df['pm25'].isnull()][['stcode', 'date']].reset_index(drop=True)
    # Convert to a list of (station, date) tuples for iteration
    missing_pairs = list(df_missing.itertuples(index=False, name=None))
    
    idw_results = []
    for target_station, target_date in tqdm(missing_pairs, desc="Filling missing values..."):
        idw_average, n_valid_neighbors = fill_missing_val(
            df_pm25 = df, 
            distance_lookup = distance_lookup,
            target_station = target_station, 
            target_date = target_date,
            return_count = True) 
        
        idw_results.append((idw_average, n_valid_neighbors, target_station, target_date))

    df['pm25_filled'] = df['pm25']  # seed with original values
    
    df_idw_results = pd.DataFrame(idw_results, columns=['pm25_idw', 'pm25_fill_n_neighbors', 'stcode', 'date'])
    
    assert df_idw_results.duplicated(subset=['stcode', 'date']).sum() == 0, \
        "Duplicate (stcode, date) entries in IDW results — check df for duplicate NaN rows"

    df = df.merge(df_idw_results, on=['stcode', 'date'], how='left')
    df['pm25_filled'] = df['pm25'].fillna(df['pm25_idw'])
    df['pm25_fill_n_neighbors'] = df['pm25_fill_n_neighbors'].astype('Int64')
    df = df.drop(columns=['pm25_idw'])
    
    return df