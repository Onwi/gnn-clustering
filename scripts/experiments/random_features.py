from pathlib import Path
import joblib
import numpy as np
from numpy.random import default_rng
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier


configs = {
    'seed': 123,
    'max_n_features': 100,
    'n_splits': 5,
    'n_permutations': 10
}
    

def run_random_feature_evaluation(X, y, path_output, configs=configs):
    rng = default_rng(configs['seed'])
    max_n_features = configs['max_n_features']
    n_splits = configs['n_splits']
    n_permutations = configs['n_permutations']
    cv = StratifiedKFold(n_splits=n_splits, shuffle=False)

    records = []
    for p in range(n_permutations):
        feature_list = rng.permutation(X.shape[1])
        for n_features in range(1, max_n_features):
            X_s = X.iloc[:, feature_list].iloc[:, :n_features]

            # model = LogisticRegression()
            model = KNeighborsClassifier()
            scores = cross_val_score(model, X_s, y, cv=cv, scoring='balanced_accuracy')
            df_current_scores = pd.DataFrame()
            df_current_scores['score'] = scores
            df_current_scores['permutation'] = p
            df_current_scores['split'] = np.arange(0, n_splits)
            df_current_scores['n_features'] = n_features
            records.append(df_current_scores)

    df_scores = pd.concat(records, axis=0, ignore_index=True)
    df_scores.to_csv(path_output)


def run_feature_rank_evaluation(feature_rank, X, y, path_output, configs=configs):
    max_n_features = configs['max_n_features']
    # max_n_features = 10
    n_splits = configs['n_splits']
    cv = StratifiedKFold(n_splits=n_splits, shuffle=False)

    records = []
    for n_features in range(1, max_n_features):
        X_s = X.iloc[:, feature_rank].iloc[:, :n_features]

        # model = LogisticRegression(class_weight='balanced')
        model = KNeighborsClassifier()
        scores = cross_val_score(model, X_s, y, cv=cv, scoring='balanced_accuracy')
        df_current_scores = pd.DataFrame()
        df_current_scores['score'] = scores
        df_current_scores['split'] = np.arange(0, n_splits)
        df_current_scores['n_features'] = n_features
        records.append(df_current_scores)

    df_scores = pd.concat(records, axis=0, ignore_index=True)
    df_scores.to_csv(path_output)
    return df_scores


def compare_with_random_features(
    path_random_features_output, 
    X,
    y
):
    path_random_features_output = Path(path_random_features_output)
    if not path_random_features_output.exists():
        run_random_feature_evaluation(X, y, path_output=path_random_features_output, configs=configs)
    
    df_random_scores = pd.read_csv(path_random_features_output)
    print(df_random_scores)


if __name__ == '__main__':
    path_df_X = '/home/thomas/Documents/Gene Expression Datasets/tcga_cohorts_and_tumor_dataset.joblib'
    path_df_y = '/home/thomas/Documents/Gene Expression Datasets/tcga_cohorts_and_tumor_dataset.csv'

    df_X = joblib.load(path_df_X)
    print(f"df_X Size: {df_X.shape}")

    df_y = pd.read_csv(path_df_y, index_col=0)
    print(df_y.head())
    print(f"Size: {df_y.shape}")

    X = df_X.transpose()
    y = df_y['sample_type']

    compare_with_random_features(
        path_random_features_output='/home/thomas/Documents/PoolingGenomicGNNs/outputs/random_features/bacc.csv',
        X=X, y=y
    )