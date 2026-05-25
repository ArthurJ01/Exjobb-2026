import os
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    f1_score,
    classification_report,
    RocCurveDisplay,
    PrecisionRecallDisplay
)

from aeon.anomaly_detection.series.distribution_based import COPOD
from aeon.anomaly_detection.series.distance_based import LOF

from deepod.models.time_series import COUTA
from deepod.models.time_series import TranAD
from deepod.models.time_series import DeepSVDDTS
from deepod.models.time_series import TcnED

import matplotlib.pyplot as plt


def evaluate_model(y_test, scores, result_path, threshold=None):
    results = {}

    roc_auc = roc_auc_score(y_test, scores)
    pr_auc = average_precision_score(y_test, scores)

    results["roc_auc"] = roc_auc
    results["pr_auc"] = pr_auc

    if threshold is not None:
        predictions = np.where(scores >= threshold, 1, 0)
        f1 = f1_score(y_test, predictions)
        results["f1"] = f1

        report_dict = classification_report(
            y_test,
            predictions,
            target_names=["Normal", "Anomaly"],
            output_dict=True
        )

        # Transpose to make metrics columns instead of rows
        df = pd.DataFrame(report_dict).transpose()
        # Add AUC to report
        df["roc_auc"] = None
        df["pr_auc"] = None

        df.loc["overall", "roc_auc"] = roc_auc
        df.loc["overall", "pr_auc"] = pr_auc
        df.loc["overall", "f1"] = f1

        df.to_csv(os.path.join(result_path,"classification_report.csv"), index=True)

    roc_auc_curve = RocCurveDisplay.from_predictions(y_test, scores)
    roc_auc_curve.figure_.savefig(os.path.join(result_path, "roc_auc_curve.png"))

    pr_auc_curve = PrecisionRecallDisplay.from_predictions(y_test, scores)
    pr_auc_curve.figure_.savefig(os.path.join(result_path, "pr_auc_curve.png"))

    return results

def cross_validate(train_df, model, params, result_path):

    split1_date = "2025-12-01"
    trainset_percentage = 0.8

    # Fold 1 df
    fold1_df = train_df[train_df['timestamp'] <= split1_date]

    fold1_train_split_idx = int(len(fold1_df) * trainset_percentage)

    fold1_train_df = fold1_df.iloc[:fold1_train_split_idx]
    fold1_val_df = fold1_df.iloc[fold1_train_split_idx:]

    # Fold 2 df
    fold2_df = train_df.iloc[::]

    fold2_train_split_idx = int(len(fold2_df) * trainset_percentage)

    fold2_train_df = fold2_df.iloc[:fold2_train_split_idx]
    fold2_val_df = fold2_df.iloc[fold2_train_split_idx:]

    print("Fold 1 start: ", fold1_df["timestamp"].iloc[0])
    print("Fold 1 end: ", fold1_df["timestamp"].iloc[-1])
    print("Fold 2 start: ", fold2_df["timestamp"].iloc[0])
    print("Fold 2 end: ", fold2_df["timestamp"].iloc[-1])

    # Fold 1 numpy
    fold1_x_train = fold1_train_df[["duration", "time_since_last_timestamp"]].values.astype(float)

    fold1_x_val = fold1_val_df[["duration", "time_since_last_timestamp"]].values.astype(float)
    fold1_y_val = fold1_val_df["label"].values.astype(float)

    # Fold 2 numpy
    fold2_x_train = fold2_train_df[["duration", "time_since_last_timestamp"]].values.astype(float)

    fold2_x_val = fold2_val_df[["duration", "time_since_last_timestamp"]].values.astype(float)
    fold2_y_val = fold2_val_df["label"].values.astype(float)

    if model == COPOD:
        fold1_model = model(**params)
        print("model 1 params: ", fold1_model.__dict__, "\n")
        fold2_model = model(**params)
        print("model 2 params: ",fold2_model.__dict__, "\n")

        fold1_model.fit(fold1_x_train, axis = 0)
        fold1_scores = fold1_model.predict(fold1_x_val, axis = 0)
        fold2_model.fit(fold2_x_train, axis = 0)
        fold2_scores = fold2_model.predict(fold2_x_val, axis = 0)

    elif model == LOF:
        fold1_model = model(**params)
        print("model 1 params: ", fold1_model.__dict__, "\n")
        fold2_model = model(**params)
        print("model 2 params: ",fold2_model.__dict__, "\n")

        fold1_scores = fold1_model.fit_predict(fold1_x_val, axis = 0)
        fold2_scores = fold2_model.fit_predict(fold2_x_val, axis = 0)

    elif model == DeepSVDDTS or model == TcnED or model == COUTA or model == TranAD:
        fold1_model = model(**params)
        print("model 1 params: ", fold1_model.__dict__, "\n")
        fold2_model = model(**params)
        print("model 2 params: ", fold2_model.__dict__, "\n")

        fold1_model.fit(fold1_x_train)
        fold1_scores = fold1_model.decision_function(fold1_x_val)
        fold2_model.fit(fold2_x_train)
        fold2_scores = fold2_model.decision_function(fold2_x_val)
    else:
        print("Unknown model")
        return None

    print("model 1 params: ", fold1_model.__dict__, "\n")
    print("model 2 params: ", fold2_model.__dict__, "\n")
    fold1_roc_auc = roc_auc_score(fold1_y_val, fold1_scores)
    fold1_pr_auc = average_precision_score(fold1_y_val, fold1_scores)

    fold2_roc_auc = roc_auc_score(fold2_y_val, fold2_scores)
    fold2_pr_auc = average_precision_score(fold2_y_val, fold2_scores)

    print(f" this is yval for fold 1: {fold1_y_val}")
    print(f" this is yval for fold 2: {fold2_y_val}")

    # Threshold calculation ###################################################################################
    fold1Precision, fold1Recall, fold1Thresholds = precision_recall_curve(fold1_y_val, fold1_scores)
    fold2Precision, fold2Recall, fold2Thresholds = precision_recall_curve(fold2_y_val, fold2_scores)

    # Removal of additional unnecessary rows (caused by precision_recall_curve function)
    fold1Precision = fold1Precision[:-1]
    fold1Recall = fold1Recall[:-1]
    fold2Precision = fold2Precision[:-1]
    fold2Recall = fold2Recall[:-1]

    F1CalcFold1_df = pd.DataFrame({
        "Precision": fold1Precision,
        "Recall": fold1Recall,
        "Thresholds": fold1Thresholds,
        "F1": None,
    })
    F1CalcFold2_df = pd.DataFrame({
        "Precision": fold2Precision,
        "Recall": fold2Recall,
        "Thresholds": fold2Thresholds,
        "F1": None,
    })

    # Calculate F1 from Precision & Recall
    # Fold 1
    for row in F1CalcFold1_df.index:
        indexRecall = F1CalcFold1_df.loc[row, "Recall"]
        indexPrecision = F1CalcFold1_df.loc[row, "Precision"]
        if indexRecall + indexPrecision == 0:
            F1CalcFold1_df.loc[row, "F1"] = 0
        else:
            F1CalcFold1_df.loc[row, "F1"] = ((2 * indexRecall * indexPrecision) / (indexRecall + indexPrecision))
    # Fold 2
    for row in F1CalcFold2_df.index:
        indexRecall = F1CalcFold2_df.loc[row, "Recall"]
        indexPrecision = F1CalcFold2_df.loc[row, "Precision"]
        if indexRecall + indexPrecision == 0:
            F1CalcFold2_df.loc[row, "F1"] = 0
        else:
            F1CalcFold2_df.loc[row, "F1"] = ((2 * indexRecall * indexPrecision) / (indexRecall + indexPrecision))

    # Find Threshold where F1 score highest (top 5)
    #  Fold 1
    F1CalcFold1_df = F1CalcFold1_df.sort_values(by='F1', ascending=False, inplace=False)
    thresholdsFold1 = F1CalcFold1_df["Thresholds"].iloc[:5]
    #  Fold 2
    F1CalcFold2_df = F1CalcFold2_df.sort_values(by='F1', ascending=False, inplace=False)
    thresholdsFold2 = F1CalcFold2_df["Thresholds"].iloc[:5]

    meanThreshold = pd.concat([thresholdsFold1, thresholdsFold2]).mean()
    # Threshold done ###################################################################################

    results = {
        "fold1_roc_auc": fold1_roc_auc,
        "fold1_pr_auc": fold1_pr_auc,
        "fold2_roc_auc": fold2_roc_auc,
        "fold2_pr_auc": fold2_pr_auc,
        "mean_threshold": meanThreshold,
    }

    df_results = pd.DataFrame([results])
    df_results.to_csv(os.path.join(result_path, "results_cross_validation.csv"), index=True)

    df_thresholds = pd.DataFrame({
        "fold1_thresholds": thresholdsFold1,
        "fold2_thresholds": thresholdsFold2
    })
    df_thresholds.to_csv(os.path.join(result_path, "thresholds_cross_validation.csv"), index=True)

    return results