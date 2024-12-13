import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import cross_val_score, GridSearchCV
import numpy as np


train_file_path = "Train_Student_Mental_Health.xlsx"
test_file_path = "Test_Student_Mental_Health.xlsx"

train_data = pd.read_excel(train_file_path)
test_data = pd.read_excel(test_file_path)


X_train = train_data.drop(columns=["Depression_Yes", "Anxiety_Yes", "Panic Attack_Yes"])
y_train_depression = train_data["Depression_Yes"]
y_train_anxiety = train_data["Anxiety_Yes"]
y_train_panic = train_data["Panic Attack_Yes"]

X_test = test_data.drop(columns=["Depression_Yes", "Anxiety_Yes", "Panic Attack_Yes"])
y_test_depression = test_data["Depression_Yes"]
y_test_anxiety = test_data["Anxiety_Yes"]
y_test_panic = test_data["Panic Attack_Yes"]


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


evaluation_results = []
cross_val_results = []


param_grids = {
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



def tune_model(model, param_grid, X_train, y_train):
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_



tuned_models = {}
targets = {
    "Depression": (y_train_depression, y_test_depression),
    "Anxiety": (y_train_anxiety, y_test_anxiety),
    "Panic Attack": (y_train_panic, y_test_panic)
}

for target, (y_train, y_test) in targets.items():
    tuned_models[target] = {}
    for model_name, param_grid in param_grids.items():
        base_model = {
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
            "KNN": KNeighborsClassifier()
        }[model_name]

        # Use scaled data if necessary
        X_train_data = X_train_scaled if model_name != "Decision Tree" else X_train

        # Tune model
        best_model = tune_model(base_model, param_grid, X_train_data, y_train)
        tuned_models[target][model_name] = best_model


# Function to train and evaluate a model
def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, model_name, target):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    report = classification_report(y_test, y_pred, output_dict=True)
    precision = report["1"]["precision"]
    recall = report["1"]["recall"]
    f1_score = report["1"]["f1-score"]
    roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else "N/A"

    evaluation_results.append({
        "Model": model_name,
        "Target": target,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1_score,
        "ROC AUC": roc_auc
    })



for target, (y_train, y_test) in targets.items():
    for model_name, model in tuned_models[target].items():
        X_train_data = X_train_scaled if model_name != "Decision Tree" else X_train
        X_test_data = X_test_scaled if model_name != "Decision Tree" else X_test
        train_and_evaluate_model(model, X_train_data, X_test_data, y_train, y_test, model_name, target)


for target_name, y_train in zip(["Depression", "Anxiety", "Panic Attack"],
                                [y_train_depression, y_train_anxiety, y_train_panic]):
    for model_name, model in tuned_models[target_name].items():
        X_data = X_train_scaled if model_name in ["Logistic Regression", "KNN"] else X_train


        scores = cross_val_score(model, X_data, y_train, cv=5, scoring='accuracy')


        cross_val_results.append({
            "Model": model_name,
            "Target": target_name,
            "Cross-Validation Mean Accuracy": np.mean(scores),
            "Cross-Validation Std Dev": np.std(scores)
        })


results_df = pd.DataFrame(evaluation_results)
cv_results_df = pd.DataFrame(cross_val_results)


output_file_eval = "Evaluation_Results_Updated.xlsx"
output_file_cv = "Evaluation_Results_With_CV.xlsx"

results_df.to_excel(output_file_eval, index=False)
cv_results_df.to_excel(output_file_cv, index=False)


print(f"Saved files: {output_file_eval}, {output_file_cv}")