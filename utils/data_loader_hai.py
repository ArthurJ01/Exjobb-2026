from sklearn.preprocessing import StandardScaler
import kagglehub
from kagglehub import KaggleDatasetAdapter
import numpy as np

def load_hai_data():
    #Train file to be used for train, Optuna and 1st and 2nd validation fold
    file_path_train_1 = "./hai-22.04/train6.csv"

    #Test file to be used for final test
    file_path_test_1 = "./hai-22.04/test4.csv"

    #Train file to be used for 3rd and 4th validation fold
    file_path_train_2 = "./hai-22.04/train5.csv"

    #Test file to be used for Optuna, 1st and 3rd validation folds
    file_path_val_1 = "./hai-22.04/test3.csv"

    #Test file to be used for 2nd and 3rd validation folds
    file_path_val_2 = "./hai-22.04/test1.csv"

    #For final model training, Optuna and 1st and 2nd validation fold
    df_train = kagglehub.dataset_load(
      KaggleDatasetAdapter.PANDAS,
      "icsdataset/hai-security-dataset",
      file_path_train_1,
      # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
    )

    #Test file to be used for final test (with df_train)
    df_test = kagglehub.dataset_load(
      KaggleDatasetAdapter.PANDAS,
      "icsdataset/hai-security-dataset",
      file_path_test_1,
      # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
    )

    #Train file to be used for 3rd and 4th validation fold (with df_train_2)
    df_train_2 = kagglehub.dataset_load(
      KaggleDatasetAdapter.PANDAS,
      "icsdataset/hai-security-dataset",
      file_path_train_2,
      # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
    )

    #Test file to be used for Optuna and 1st validation fold (with df_train)
    df_val_1_1 = kagglehub.dataset_load(
      KaggleDatasetAdapter.PANDAS,
      "icsdataset/hai-security-dataset",
      file_path_val_1,
      # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
    )
    #Test file to be used for 3rd validation fold (with df_train_2)
    df_val_1_2 = df_val_1_1.copy()

    #Test file to be used for 2nd validation fold (with df_train)
    df_val_2_1 = kagglehub.dataset_load(
      KaggleDatasetAdapter.PANDAS,
      "icsdataset/hai-security-dataset",
      file_path_val_2,
      # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
    )

    #Test file to be used for 4th validation fold (with df_train_2)
    df_val_2_2 = df_val_2_1.copy()

    timestamp_column = "timestamp"
    label_column = "Attack"

    #Labels for test set
    y_test_1 = df_test[label_column].astype(int).to_numpy()
    #Labels for Optuna, 1st and 3rd validation fold
    y_val_1 = df_val_1_1[label_column].astype(int).to_numpy()
    #Labels for 2nd and 4th validation fold
    y_val_2 = df_val_2_2[label_column].astype(int).to_numpy()

    feature_columns = [
        column for column in df_train.columns
        if column not in [timestamp_column, label_column]
    ]

    #scaler for df_train
    scaler_1 = StandardScaler()
    #scaler for df_train_2
    scaler_2 = StandardScaler()

    x_train = scaler_1.fit_transform(df_train[feature_columns]).astype(np.float32)
    x_test = scaler_1.transform(df_test[feature_columns]).astype(np.float32)

    x_val_1_1 = scaler_1.transform(df_val_1_1[feature_columns]).astype(np.float32)
    x_val_2_1 = scaler_1.transform(df_val_2_1[feature_columns]).astype(np.float32)

    x_train_2 = scaler_2.fit_transform(df_train_2[feature_columns]).astype(np.float32)
    x_val_1_2 = scaler_2.transform(df_val_1_2[feature_columns]).astype(np.float32)
    x_val_2_2 = scaler_2.transform(df_val_2_2[feature_columns]).astype(np.float32)
    # For final train and test use: x_train, x_test, y_test_1
    # For Optuna use: x_train, x_val_1_1, y_val_1
    # For 1st validation fold use: x_train, x_val_1_1, y_val_1
    # For 2nd validation fold use: x_train, x_val_2_1, y_val_2
    # For 3rd validation fold use: x_train_2, x_val_1_2, y_val_1
    # For 4th validation fold use: x_train_2, x_val_2_2, y_val_2
    return x_train, x_test, y_test_1, x_val_1_1, x_val_2_1, y_val_1, x_train_2, x_val_1_2, x_val_2_2, y_val_2

# Helper functions to hide the trash
def load_hai_train():
    x_train, x_test, y_test_1, x_val_1_1, x_val_2_1, y_val_1, x_train_2, x_val_1_2, x_val_2_2, y_val_2 = load_hai_data()
    return x_train, x_test, y_test_1

def load_hai_optuna():
    x_train, x_test, y_test_1, x_val_1_1, x_val_2_1, y_val_1, x_train_2, x_val_1_2, x_val_2_2, y_val_2 = load_hai_data()
    return x_train, x_val_1_1, y_val_1