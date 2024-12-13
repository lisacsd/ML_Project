import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.model_selection import cross_val_score, GridSearchCV
import numpy as np


train_data_file = "Train_Student_Mental_Health.xlsx"
test_data_file = "Test_Student_Mental_Health.xlsx"

train_dataset = pd.read_excel(train_data_file)
test_dataset = pd.read_excel(test_data_file)

X_train_features = train_dataset.drop(columns=["Depression_Yes", "Anxiety_Yes", "Panic Attack_Yes"])
y_train_depr = train_dataset["Depression_Yes"]
y_train_anx = train_dataset["Anxiety_Yes"]
y_train_panic = train_dataset["Panic Attack_Yes"]

X_test_features = test_dataset.drop(columns=["Depression_Yes", "Anxiety_Yes", "Panic Attack_Yes"])
y_test_depr = test_dataset["Depression_Yes"]
y_test_anx = test_dataset["Anxiety_Yes"]
y_test_panic = test_dataset["Panic Attack_Yes"]


data_scaler = StandardScaler()
X_train_scaled_features = data_scaler.fit_transform(X_train_features)
X_test_scaled_features = data_scaler.transform(X_test_features)


eval_results = []
cross_validation_results = []

model_param_grids = {
    "Decision Tree": {
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    "Logistic Regression": {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'lbfgs']
    },
    "KNN": {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance']
    }
}


def optimize_model(model, param_grid, X_train, y_train):
    model_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    model_search.fit(X_train, y_train)
    return model_search.best_estimator_


trained_models = {}
prediction_targets = {
    "Depression": (y_train_depr, y_test_depr),
    "Anxiety": (y_train_anx, y_test_anx),
    "Panic Attack": (y_train_panic, y_test_panic)
}


for target_label, (train_target, test_target) in prediction_targets.items():
    trained_models[target_label] = {}
    for model_name, grid_params in model_param_grids.items():
        base_model = {
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
            "KNN": KNeighborsClassifier()
        }[model_name]

        X_train_data = X_train_scaled_features if model_name != "Decision Tree" else X_train_features
        best_trained_model = optimize_model(base_model, grid_params, X_train_data, train_target)
        trained_models[target_label][model_name] = best_trained_model


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name, target_label):
    model.fit(X_train, y_train)
    y_predictions = model.predict(X_test)
    y_probabilities = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    test_accuracy = accuracy_score(y_test, y_predictions)
    report = classification_report(y_test, y_predictions, output_dict=True)
    precision_score = report["1"]["precision"]
    recall_score = report["1"]["recall"]
    f1_score_metric = report["1"]["f1-score"]
    roc_auc_metric = roc_auc_score(y_test, y_probabilities) if y_probabilities is not None else "N/A"

    eval_results.append({
        "Model": model_name,
        "Target": target_label,
        "Accuracy": test_accuracy,
        "Precision": precision_score,
        "Recall": recall_score,
        "F1-Score": f1_score_metric,
        "ROC AUC": roc_auc_metric
    })

for target_label, (train_target, test_target) in prediction_targets.items():
    for model_name, trained_model in trained_models[target_label].items():
        X_train_data = X_train_scaled_features if model_name != "Decision Tree" else X_train_features
        X_test_data = X_test_scaled_features if model_name != "Decision Tree" else X_test_features
        evaluate_model(trained_model, X_train_data, X_test_data, train_target, test_target, model_name, target_label)


for target_label, train_target in zip(["Depression", "Anxiety", "Panic Attack"],
                                      [y_train_depr, y_train_anx, y_train_panic]):
    for model_name, trained_model in trained_models[target_label].items():
        X_train_data = X_train_scaled_features if model_name in ["Logistic Regression", "KNN"] else X_train_features
        cv_scores = cross_val_score(trained_model, X_train_data, train_target, cv=5, scoring='accuracy')
        cross_validation_results.append({
            "Model": model_name,
            "Target": target_label,
            "CV Mean Accuracy": np.mean(cv_scores),
            "CV Std Dev": np.std(cv_scores)
        })


eval_results_df = pd.DataFrame(eval_results)
cv_results_df = pd.DataFrame(cross_validation_results)

eval_file = "Evaluation_Results_Updated.xlsx"
cv_file = "Evaluation_Results_With_CV.xlsx"

eval_results_df.to_excel(eval_file, index=False)
cv_results_df.to_excel(cv_file, index=False)

print(f"Saved files: {eval_file}, {cv_file}")