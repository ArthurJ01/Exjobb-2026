import os
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)

from aeon.anomaly_detection.series.distribution_based import COPOD
from aeon.anomaly_detection.series.distance_based import LOF

from deepod.models.time_series import COUTA
from deepod.models.time_series import TranAD
from deepod.models.time_series import DeepSVDDTS
from deepod.models.time_series import TcnED

from utils.data_loader_hai import load_hai_data

def cross_validate_hai(model, params, result_path):
    # For 1st validation fold use: x_train, x_val_1_1, y_val_1
    # For 2nd validation fold use: x_train, x_val_2_1, y_val_2
    # For 3rd validation fold use: x_train_2, x_val_1_2, y_val_1
    # For 4th validation fold use: x_train_2, x_val_2_2, y_val_2
    x_train, x_test, y_test_1, x_val_1_1, x_val_2_1, y_val_1, x_train_2, x_val_1_2, x_val_2_2, y_val_2 = load_hai_data()


    if model == COPOD:
        fold1_model = model(**params)
        print("model 1 params: ", fold1_model.__dict__, "\n")
        fold1_model.fit(x_train, axis=0)
        fold1_scores = fold1_model.predict(x_val_1_1, axis=0)
        del fold1_model

        fold2_model = model(**params)
        print("model 2 params: ",fold2_model.__dict__, "\n")
        fold2_model.fit(x_train, axis=0)
        fold2_scores = fold2_model.predict(x_val_2_1, axis=0)
        del fold2_model

        fold3_model = model(**params)
        print("model 3 params: ", fold3_model.__dict__, "\n")
        fold3_model.fit(x_train_2, axis=0)
        fold3_scores = fold3_model.predict(x_val_1_2, axis=0)
        del fold3_model

        fold4_model = model(**params)
        print("model 4 params: ", fold4_model.__dict__, "\n")
        fold4_model.fit(x_train_2, axis=0)
        fold4_scores = fold4_model.predict(x_val_2_2, axis=0)
        del fold4_model

    elif model == LOF:
        fold1_model = model(**params)
        print("model 1 params: ", fold1_model.__dict__, "\n")
        fold1_scores = fold1_model.fit_predict(x_val_1_1, axis=0)
        del fold1_model

        fold2_model = model(**params)
        print("model 2 params: ", fold2_model.__dict__, "\n")
        fold2_scores = fold2_model.fit_predict(x_val_2_1, axis=0)
        del fold2_model

        fold3_model = model(**params)
        print("model 3 params: ", fold3_model.__dict__, "\n")
        fold3_scores = fold3_model.fit_predict(x_val_1_2, axis=0)
        del fold3_model

        fold4_model = model(**params)
        print("model 4 params: ", fold4_model.__dict__, "\n")
        fold4_scores = fold4_model.fit_predict(x_val_2_2, axis=0)
        del fold4_model

    elif model == DeepSVDDTS or model == TcnED or model == COUTA or model == TranAD:
        fold1_model = model(**params)
        print("model 1 params: ", fold1_model.__dict__, "\n")
        fold2_model = model(**params)
        print("model 2 params: ", fold2_model.__dict__, "\n")
        fold3_model = model(**params)
        print("model 3 params: ", fold3_model.__dict__, "\n")
        fold4_model = model(**params)
        print("model 4 params: ", fold4_model.__dict__, "\n")

        fold1_model.fit(x_train)
        fold1_scores = fold1_model.decision_function(x_val_1_1)
        fold2_model.fit(x_train)
        fold2_scores = fold2_model.decision_function(x_val_2_1)
        fold3_model.fit(x_train_2)
        fold3_scores = fold3_model.decision_function(x_val_1_2)
        fold4_model.fit(x_train_2)
        fold4_scores = fold4_model.decision_function(x_val_2_2)
    else:
        print("Unknown model")
        return None

    fold1_roc_auc = roc_auc_score(y_val_1, fold1_scores)
    fold1_pr_auc = average_precision_score(y_val_1, fold1_scores)

    fold2_roc_auc = roc_auc_score(y_val_2, fold2_scores)
    fold2_pr_auc = average_precision_score(y_val_2, fold2_scores)

    fold3_roc_auc = roc_auc_score(y_val_1, fold3_scores)
    fold3_pr_auc = average_precision_score(y_val_1, fold3_scores)

    fold4_roc_auc = roc_auc_score(y_val_2, fold4_scores)
    fold4_pr_auc = average_precision_score(y_val_2, fold4_scores)

    # Threshold calculation ###################################################################################
    fold1Precision, fold1Recall, fold1Thresholds = precision_recall_curve(y_val_1, fold1_scores)
    fold2Precision, fold2Recall, fold2Thresholds = precision_recall_curve(y_val_2, fold2_scores)
    fold3Precision, fold3Recall, fold3Thresholds = precision_recall_curve(y_val_1, fold3_scores)
    fold4Precision, fold4Recall, fold4Thresholds = precision_recall_curve(y_val_2, fold4_scores)

    # Removal of additional unnecessary rows (caused by precision_recall_curve function)
    fold1Precision = fold1Precision[:-1]
    fold1Recall = fold1Recall[:-1]
    fold2Precision = fold2Precision[:-1]
    fold2Recall = fold2Recall[:-1]
    fold3Precision = fold3Precision[:-1]
    fold3Recall = fold3Recall[:-1]
    fold4Precision = fold4Precision[:-1]
    fold4Recall = fold4Recall[:-1]

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

    F1CalcFold3_df = pd.DataFrame({
        "Precision": fold3Precision,
        "Recall": fold3Recall,
        "Thresholds": fold3Thresholds,
        "F1": None,
    })

    F1CalcFold4_df = pd.DataFrame({
        "Precision": fold4Precision,
        "Recall": fold4Recall,
        "Thresholds": fold4Thresholds,
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

    # Fold 3
    for row in F1CalcFold3_df.index:
        indexRecall = F1CalcFold3_df.loc[row, "Recall"]
        indexPrecision = F1CalcFold3_df.loc[row, "Precision"]
        if indexRecall + indexPrecision == 0:
            F1CalcFold3_df.loc[row, "F1"] = 0
        else:
            F1CalcFold3_df.loc[row, "F1"] = ((2 * indexRecall * indexPrecision) / (indexRecall + indexPrecision))

    # Fold 4
    for row in F1CalcFold4_df.index:
        indexRecall = F1CalcFold4_df.loc[row, "Recall"]
        indexPrecision = F1CalcFold4_df.loc[row, "Precision"]
        if indexRecall + indexPrecision == 0:
            F1CalcFold4_df.loc[row, "F1"] = 0
        else:
            F1CalcFold4_df.loc[row, "F1"] = (
                        (2 * indexRecall * indexPrecision) / (indexRecall + indexPrecision))

    # Find Threshold where F1 score highest (top 5)
    #  Fold 1
    F1CalcFold1_df = F1CalcFold1_df.sort_values(by='F1', ascending=False, inplace=False)
    thresholdsFold1 = F1CalcFold1_df["Thresholds"].iloc[:5]
    #  Fold 2
    F1CalcFold2_df = F1CalcFold2_df.sort_values(by='F1', ascending=False, inplace=False)
    thresholdsFold2 = F1CalcFold2_df["Thresholds"].iloc[:5]

    F1CalcFold3_df = F1CalcFold3_df.sort_values(by='F1', ascending=False, inplace=False)
    thresholdsFold3 = F1CalcFold3_df["Thresholds"].iloc[:5]

    F1CalcFold4_df = F1CalcFold4_df.sort_values(by='F1', ascending=False, inplace=False)
    thresholdsFold4 = F1CalcFold4_df["Thresholds"].iloc[:5]

    meanThreshold = pd.concat([thresholdsFold1, thresholdsFold2, thresholdsFold3, thresholdsFold4]).mean()
    # Threshold done ###################################################################################

    results = {
        "fold1_roc_auc": fold1_roc_auc,
        "fold1_pr_auc": fold1_pr_auc,
        "fold2_roc_auc": fold2_roc_auc,
        "fold2_pr_auc": fold2_pr_auc,
        "fold3_roc_auc": fold3_roc_auc,
        "fold3_pr_auc": fold3_pr_auc,
        "fold4_roc_auc": fold4_roc_auc,
        "fold4_pr_auc": fold4_pr_auc,
        "mean_threshold": meanThreshold,
    }

    df_results = pd.DataFrame([results])
    df_results.to_csv(os.path.join(result_path, "results_cross_validation.csv"), index=True)

    df_thresholds = pd.DataFrame({
        "fold1_thresholds": thresholdsFold1,
        "fold2_thresholds": thresholdsFold2,
        "fold3_thresholds": thresholdsFold3,
        "fold4_thresholds": thresholdsFold4
    })
    df_thresholds.to_csv(os.path.join(result_path, "thresholds_cross_validation.csv"), index=True)

    return results