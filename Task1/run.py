import numpy as np
from pathlib import Path

from advanced_models import improved_ridge_regression

DATA_DIR = Path("dataset")

x_train_raw = np.genfromtxt(DATA_DIR / "x_train.csv", delimiter=",", skip_header=1)
y_train_raw = np.genfromtxt(DATA_DIR / "y_train.csv", delimiter=",", skip_header=1)
x_test_raw = np.genfromtxt(DATA_DIR / "x_test.csv", delimiter=",", skip_header=1)

x_train = x_train_raw[:, 1:]
y_train = y_train_raw[:, 1]
x_test = x_test_raw[:, 1:]
test_ids = x_test_raw[:, 0].astype(int)

# Match the final notebook pipeline.
technical_features = {
    "SEQNO", "_PSU", "FMONTH", "IDATE", "IMONTH", "IDAY", "IYEAR",
    "DISPCODE", "_STSTR", "_STRWT", "_RAWRAKE", "_WT2RAKE", "_CLLCPWT",
    "_DUALCOR", "_LLCPWT", "_MICHD",
}
low_information_features = {
    "FRUTDA1_", "LENGEXAM", "_FRUTSUM", "_FRT16", "VINOCRE2", "VIINSUR2",
    "HIVTST6", "_AIDTST3", "VIEYEXM2", "_PRACE1", "_MRACE1", "EXERHMM1",
    "_RACEGR3", "_RACE_G1", "_MINAC11",
}
feature_names = (DATA_DIR / "x_train.csv").read_text(encoding="utf-8").splitlines()[0].split(",")[1:]
features_to_remove = technical_features | low_information_features
kept_indices = [i for i, name in enumerate(feature_names) if name not in features_to_remove]
x_train = x_train[:, kept_indices]
x_test = x_test[:, kept_indices]
feature_names = [feature_names[i] for i in kept_indices]
feature_index = {name: i for i, name in enumerate(feature_names)}

x_train_engineered = x_train.copy()
x_test_engineered = x_test.copy()
log_features = {
    "WEIGHT2", "HEIGHT3", "AVEDRNK2", "MAXDRNKS", "FRUIT1", "FVGREEN",
    "VEGETAB1", "EXEROFT2", "EXERHMM2", "FLSHTMY2", "HIVTSTD3", "BLDSUGAR",
    "FEETCHK2", "DOCTDIAB", "CHKHEMO3", "FEETCHK", "DROCDY3_", "_DRNKWEK",
    "MAXVO2_", "FC60_", "PADUR1_", "PADUR2_", "PAFREQ1_", "_MINAC21",
    "PAMIN11_", "PAMIN21_", "PA1MIN_", "PAVIG11_", "PAVIG21_", "PA1VIGM_",
}
stretched_features = {"LONGWTCH", "HTIN4", "HTM4"}
for name in log_features:
    if name in feature_index:
        i = feature_index[name]
        x_train_engineered[:, i] = np.where(np.isnan(x_train[:, i]), np.nan, np.log1p(x_train[:, i]))
        x_test_engineered[:, i] = np.where(np.isnan(x_test[:, i]), np.nan, np.log1p(x_test[:, i]))
for name in stretched_features:
    if name in feature_index:
        i = feature_index[name]
        observed = x_train[:, i][~np.isnan(x_train[:, i])]
        median = np.median(observed)
        iqr = np.percentile(observed, 75) - np.percentile(observed, 25)
        x_train_engineered[:, i] = np.where(np.isnan(x_train[:, i]), np.nan, (x_train[:, i] - median) / iqr)
        x_test_engineered[:, i] = np.where(np.isnan(x_test[:, i]), np.nan, (x_test[:, i] - median) / iqr)

x_train_final = x_train_engineered.copy()
x_test_final = x_test_engineered.copy()

def add_feature(train_values, test_values):
    global x_train_final, x_test_final
    x_train_final = np.c_[x_train_final, train_values]
    x_test_final = np.c_[x_test_final, test_values]

train_missing_count = np.isnan(x_train_engineered).sum(axis=1).astype(float)
test_missing_count = np.isnan(x_test_engineered).sum(axis=1).astype(float)
add_feature(train_missing_count, test_missing_count)
add_feature(train_missing_count / x_train_engineered.shape[1], test_missing_count / x_test_engineered.shape[1])

def get_feature(data, name):
    if name in feature_index:
        return data[:, feature_index[name]]
    return np.full(data.shape[0], np.nan)

train_age = get_feature(x_train_engineered, "_AGEG5YR")
test_age = get_feature(x_test_engineered, "_AGEG5YR")
train_bmi = get_feature(x_train_engineered, "_BMI5") / 100.0
test_bmi = get_feature(x_test_engineered, "_BMI5") / 100.0
train_hypertension = (get_feature(x_train_engineered, "_RFHYPE5") == 1).astype(float)
test_hypertension = (get_feature(x_test_engineered, "_RFHYPE5") == 1).astype(float)
train_smoker = (get_feature(x_train_engineered, "_RFSMOK3") == 1).astype(float)
test_smoker = (get_feature(x_test_engineered, "_RFSMOK3") == 1).astype(float)
add_feature(train_age**2, test_age**2)
add_feature(train_bmi**2, test_bmi**2)
add_feature(np.nan_to_num(train_age) * train_hypertension, np.nan_to_num(test_age) * test_hypertension)
add_feature(np.nan_to_num(train_age) * train_smoker, np.nan_to_num(test_age) * test_smoker)
add_feature(np.nan_to_num(train_bmi) * train_hypertension, np.nan_to_num(test_bmi) * test_hypertension)

def comorbidity_score(data):
    conditions = [
        get_feature(data, "_RFHYPE5") == 1,
        get_feature(data, "_RFCHOL") == 1,
        np.isin(get_feature(data, "DIABETE3"), [1, 2, 3]),
        get_feature(data, "CVDSTRK3") == 1,
        get_feature(data, "CHCCOPD1") == 1,
        get_feature(data, "HAVARTH3") == 1,
        get_feature(data, "CHCKIDNY") == 1,
    ]
    return np.sum(np.vstack(conditions), axis=0).astype(float)

train_comorbidity = comorbidity_score(x_train_engineered)
test_comorbidity = comorbidity_score(x_test_engineered)
add_feature(train_comorbidity, test_comorbidity)
for name in ("BPMEDS", "DIABAGE2", "CDHELP"):
    if name in feature_index:
        add_feature(np.isnan(get_feature(x_train_engineered, name)).astype(float), np.isnan(get_feature(x_test_engineered, name)).astype(float))
add_feature(np.nan_to_num(train_age) * np.nan_to_num(train_bmi), np.nan_to_num(test_age) * np.nan_to_num(test_bmi))
add_feature(np.nan_to_num(train_age) * train_comorbidity, np.nan_to_num(test_age) * test_comorbidity)
add_feature(train_hypertension * train_smoker, test_hypertension * test_smoker)
add_feature(train_comorbidity**2, test_comorbidity**2)
add_feature(
    np.nan_to_num(train_age) * np.nan_to_num(get_feature(x_train_engineered, "_RFHLTH")),
    np.nan_to_num(test_age) * np.nan_to_num(get_feature(x_test_engineered, "_RFHLTH")),
)

feature_means = np.nanmean(x_train_final, axis=0)
if np.isnan(feature_means).any():
    raise ValueError("At least one final feature column contains only NaN values")
x_train_final = np.where(np.isnan(x_train_final), feature_means, x_train_final)
x_test_final = np.where(np.isnan(x_test_final), feature_means, x_test_final)
mean_x = np.mean(x_train_final, axis=0)
std_x = np.std(x_train_final, axis=0)
std_x[std_x == 0] = 1.0
x_train_final = np.c_[np.ones(x_train_final.shape[0]), (x_train_final - mean_x) / std_x]
x_test_final = np.c_[np.ones(x_test_final.shape[0]), (x_test_final - mean_x) / std_x]

rng = np.random.default_rng(42)
permutation = rng.permutation(len(y_train))
split_point = int(0.8 * len(y_train))
fit_rows = permutation[:split_point]
validation_rows = permutation[split_point:]
validation_weights, _ = improved_ridge_regression(
    y_train[fit_rows],
    x_train_final[fit_rows],
    lambda_=0.001,
)
validation_scores = x_train_final[validation_rows] @ validation_weights

threshold_candidates = np.unique(
    np.r_[0.0, np.percentile(validation_scores, np.linspace(5.0, 95.0, 19))]
)
best_threshold = 0.0
best_balanced_accuracy = -np.inf
for candidate in threshold_candidates:
    validation_predictions = np.where(validation_scores >= candidate, 1, -1)
    positive_mask = y_train[validation_rows] == 1
    negative_mask = y_train[validation_rows] == -1
    recall_positive = np.mean(validation_predictions[positive_mask] == 1)
    recall_negative = np.mean(validation_predictions[negative_mask] == -1)
    candidate_score = 0.5 * (recall_positive + recall_negative)
    if candidate_score > best_balanced_accuracy:
        best_balanced_accuracy = candidate_score
        best_threshold = candidate

weights, loss = improved_ridge_regression(y_train, x_train_final, lambda_=0.001)
test_scores = x_test_final @ weights
predictions = np.where(test_scores >= best_threshold, 1, -1)

submission = np.column_stack((test_ids, predictions))
np.savetxt(
    "submission.csv",
    submission,
    delimiter=",",
    header="Id,Prediction",
    comments="",
    fmt="%d",
)

print(f"Training loss: {loss:.6f}")
print(f"Validation threshold: {best_threshold:.6f}")
print(f"Validation balanced accuracy: {best_balanced_accuracy:.4%}")
print(f"Saved {len(predictions)} predictions to submission.csv")
