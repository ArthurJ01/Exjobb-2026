import pandas as pd
import numpy as np
'''
Replaces this in every notebook

train_df = pd.read_csv("Dataset/Processed Dataset/volvoTrain.csv")
test_df = pd.read_csv("Dataset/Processed Dataset/volvoTest.csv")

timestamp_col = "timestamp"
feature_cols = ["duration", "time_since_last_timestamp"]
label_col = "label"

timestamps_train = train_df["timestamp"].values
timestamps_test = test_df["timestamp"].values

X_train = train_df[feature_cols].values.astype(float)
X_test = test_df[feature_cols].values.astype(float)

Y_train = train_df[label_col].values.astype(float)
Y_test = test_df[label_col].values.astype(float)

'''
def load_volvo(train_path, test_path,
               timestamp_col="timestamp",
               feature_cols=None,
               label_col="label"):

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if feature_cols is None:
        feature_cols = ["duration", "time_since_last_timestamp"]
    else:
        feature_cols = feature_cols

    x_train = train_df[feature_cols].values.astype(float)
    x_test  = test_df[feature_cols].values.astype(float)

    y_train = train_df[label_col].values.astype(float)
    y_test  = test_df[label_col].values.astype(float)

    timestamps_train = train_df[timestamp_col].values
    timestamps_test  = test_df[timestamp_col].values

    return x_train, y_train, x_test, y_test, timestamps_train, timestamps_test, train_df, test_df

def load_optuna_val(train_df):
    split_start_date = "2025-12-07"
    trainset_percentage = 0.8

    # Fold 1 df
    optuna_df = train_df[train_df['timestamp'] >= split_start_date]

    print("Optuna start: ", optuna_df["timestamp"].iloc[0])
    print("Optuna end: ", optuna_df["timestamp"].iloc[-1])

    return optuna_df