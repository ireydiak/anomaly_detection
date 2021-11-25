from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import category_encoders as ce
import warnings
import utils

warnings.filterwarnings('ignore')

NORMAL_LABEL = 0
ANORMAL_LABEL = 1

dataset_name = 'cicids2017'

COLS_TO_DROP = [
    'Dst IP',
    'Destination Port'
    'Flow ID',
    'Src IP',
    'Src Port',
    'Flow Duration',
    'Protocol',
    'Timestamp',
]

COLS = [
    'Dst Port',
    'Tot Fwd Pkts',
    'Tot Bwd Pkts',
    'TotLen Fwd Pkts',
    'TotLen Bwd Pkts',
    'Fwd Pkt Len Max',
    'Fwd Pkt Len Min',
    'Fwd Pkt Len Mean',
    'Fwd Pkt Len Std',
    'Bwd Pkt Len Max',
    'Bwd Pkt Len Min',
    'Bwd Pkt Len Mean',
    'Bwd Pkt Len Std',
    'Flow Byts/s',
    'Flow Pkts/s',
    'Flow IAT Mean',
    'Flow IAT Std',
    'Flow IAT Max',
    'Flow IAT Min',
    'Fwd IAT Tot',
    'Fwd IAT Mean',
    'Fwd IAT Std',
    'Fwd IAT Max',
    'Fwd IAT Min',
    'Bwd IAT Tot',
    'Bwd IAT Mean',
    'Bwd IAT Std',
    'Bwd IAT Max',
    'Bwd IAT Min',
    'Fwd PSH Flags',
    'Fwd URG Flags',
    'Fwd Header Len',
    'Bwd Header Len',
    'Fwd Pkts/s',
    'Bwd Pkts/s',
    'Pkt Len Min',
    'Pkt Len Max',
    'Pkt Len Mean',
    'Pkt Len Std',
    'Pkt Len Var',
    'FIN Flag Cnt',
    'SYN Flag Cnt',
    'RST Flag Cnt',
    'PSH Flag Cnt',
    'ACK Flag Cnt',
    'URG Flag Cnt',
    'CWE Flag Cnt',
    'ECE Flag Cnt',
    'Down/Up Ratio',
    'Pkt Size Avg',
    'Fwd Seg Size Avg',
    'Bwd Seg Size Avg',
    'Subflow Fwd Pkts',
    'Subflow Fwd Byts',
    'Subflow Bwd Pkts',
    'Subflow Bwd Byts',
    'Fwd Act Data Pkts',
    'Fwd Seg Size Min',
    'Active Mean',
    'Active Std',
    'Active Max',
    'Active Min',
    'Idle Mean',
    'Idle Std',
    'Idle Max',
    'Idle Min'
]


def drop_uniq_cols(df: pd.DataFrame):
    uniq_cols = df.columns[df.nunique() == 1]
    df.drop(columns=uniq_cols)
    return df, list(map(lambda col: col, df.columns[uniq_cols]))


def clean_step(path_to_file: str, nan_thresh: float = 0.1, negative_thresh: float = 0.1) -> (pd.DataFrame, dict):
    total_rows, deleted_rows, deleted_features = 0, 0, len(COLS_TO_DROP)
    chunks, n_features = [], []
    stats = defaultdict()

    print(f"Cleaning file {path_to_file}")
    chunk = pd.read_csv(path_to_file)
    chunk.rename(columns=lambda x: x.strip(), inplace=True)
    n_features.append(len(chunk.columns))
    total_rows += len(chunk)
    # Drop target columns if they exist
    chunk.drop(columns=COLS_TO_DROP, errors='ignore', inplace=True)

    # Drop columns with unique values
    print("Dropping columns with unique values...")
    uniq_cols = chunk.columns[chunk.nunique() == 1]
    chunk.drop(columns=uniq_cols, inplace=True)
    deleted_features += len(uniq_cols)
    stats["uniq_cols"] = '; '.join(list(uniq_cols))
    print("Dropped {}".format(', '.join(uniq_cols)))

    # Drop rows with NaN or invalid (INF) values
    # If the number of rows is greater than nan_thresh, then we remove the feature instead.
    print("Filtering NaN/INF values...")
    chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
    nan_cols = chunk.columns[chunk.isna().any()]
    stats["dropped_nan_cols"] = []
    for col in nan_cols:
        cnt = chunk[col].isna().sum()
        stats[col + " NAN CNT"] = cnt
        if cnt / len(chunk[col]) >= nan_thresh:
            chunk.drop(columns=[col], inplace=True)
            deleted_features += 1
            stats["dropped_nan_cols"].append(col)
        else:
            deleted_rows += cnt
            chunk[col].dropna(inplace=True)
            assert len(chunk) < len(chunk) + cnt
    print("Dropped columns {}".format(', '.join(stats["dropped_nan_cols"])))
    stats["dropped_nan_cols"] = '; '.join(stats["dropped_nan_cols"])

    # Filtering negative values
    print("Filtering negative values...")
    num_cols = chunk.select_dtypes(exclude=["object", "category"]).columns
    neg_cols = num_cols[(chunk[num_cols] < 0).any()]
    stats["dropped_negative_cols"] = []
    for col in neg_cols:
        cnt = (chunk[col] < 0).sum()
        if cnt == 0:
            continue
        stats[col + " NEG CNT"] = cnt
        if cnt / len(chunk[col]) >= negative_thresh:
            chunk.drop(columns=[col], inplace=True)
            deleted_features += 1
            stats["dropped_negative_cols"].append(col)
        else:
            deleted_rows += cnt
            chunk.drop(chunk[col][chunk[col] < 0].index, inplace=True)
            assert len(chunk) < len(chunk) + cnt
    print("Dropped columns {}".format(', '.join(stats["dropped_negative_cols"])))
    stats["dropped_negative_cols"] = '; '.join(stats["dropped_negative_cols"])

    # Converting labels to binary values
    chunk['Label'] = chunk['Label'].apply(lambda x: NORMAL_LABEL if x == 'BENIGN' else ANORMAL_LABEL)

    # Adding chunk to chunks
    chunks.append(chunk)

    df = chunk.dropna()
    tot_features = np.argmax(np.bincount(n_features))
    final_features_len = len(df.columns)
    stats = dict({
        "Total Rows": str(total_rows),
        "Dropped Rows": str(deleted_rows),
        "Final Rows": str(total_rows - deleted_rows),
        "Ratio": f"{(deleted_rows / total_rows):1.4f}",
        "Total Features": str(tot_features),
        "Dropped Features": str(tot_features - final_features_len),
        "Final Features": str(final_features_len),
    }, **stats)
    return df, stats


def normalize_step(df: pd.DataFrame, cols: list, base_path: str, fname: str):
    print(f'Processing {len(cols)} features for {fname}')
    # Preprocessing inspired by https://journalofbigdata.springeropen.com/articles/10.1186/s40537-021-00426-w
    # Split numerical and non-numerical columns
    num_cols = df[cols].select_dtypes(exclude=["object", "category"]).columns.tolist()
    cat_cols = df[cols].select_dtypes(include=["category", "object"]).columns.tolist()
    # Optinally handle categorical values
    if cat_cols:
        print("Converting categorical attributes {} to numeric".format(', '.join(cat_cols)))
        perm = np.random.permutation(len(df))
        X = df.iloc[perm].reset_index(drop=True)
        y_prime = df['Label'].iloc[perm].reset_index(drop=True)
        enc = ce.CatBoostEncoder(verbose=1, cols=cat_cols)
        df = enc.fit_transform(X, y_prime)
    # Keep labels aside
    y = df['Label'].to_numpy()
    # Keep only a subset of the features
    df = df[cols]
    # Normalize numerical data
    scaler = MinMaxScaler()
    # Select numerical columns with values in the range (0, 1)
    # This way we avoid normalizing values that are already between 0 and 1.
    to_scale = df[num_cols][(df[num_cols] < 0.0).any(axis=1) & (df[num_cols] > 1.0).any(axis=1)].columns
    print(f'Scaling {len(to_scale)} columns')
    df[to_scale] = scaler.fit_transform(df[to_scale].values.astype(np.float64))
    # Merge normalized dataframe with labels
    X = np.concatenate(
        (df.values, y.reshape(-1, 1)),
        axis=1
    )
    del df
    np.savez(f'{base_path}/{utils.folder_struct["minify_step"]}/{fname}.npz', ids2017=X.astype(np.float64))
    print(f'Saved {base_path}/{fname}.npz')


if __name__ == '__main__':
    # Assumes `path` points to the location of the original CSV files.
    # `path` must only contain CSV files and not other file types such as folders. 
    path, export_path = utils.parse_args()
    # 0 - Prepare folder structure
    utils.prepare(export_path)
    # 1 - Clean the data (remove invalid rows and columns)
    df, clean_stats = clean_step(path)
    # Save info about cleaning step
    utils.save_stats(export_path + '/{}_info.csv'.format(dataset_name), clean_stats)

    cols = list(df.columns)
    # 2 - Normalize numerical values and treat categorical values
    to_process = [
        (list(set(cols) - {'Label'}), 'feature_group_5'),
        (list(set(cols) - {'Dst Port', 'Label'}), 'feature_group_5A'),
    ]
    df['Dst Port'] = df['Dst Port'].astype('category')
    for features, fname in to_process:
        normalize_step(df, features, export_path, fname)
