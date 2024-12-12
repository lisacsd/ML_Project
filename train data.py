# Import libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.model_selection import GridSearchCV

# Load the training and testing datasets
train_file_path = "Train_Student_Mental_Health.xlsx"
test_file_path = "Test_Student_Mental_Health.xlsx"

train_data = pd.read_excel(train_file_path)
test_data = pd.read_excel(test_file_path)

# Separate features and target variables
X_train = train_data.drop(columns=["Depression_Yes", "Anxiety_Yes", "Panic Attack_Yes"])
y_train_depression = train_data["Depression_Yes"]
y_train_anxiety = train_data["Anxiety_Yes"]
y_train_panic = train_data["Panic Attack_Yes"]

X_test = test_data.drop(columns=["Depression_Yes", "Anxiety_Yes", "Panic Attack_Yes"])
y_test_depression = test_data["Depression_Yes"]
y_test_anxiety = test_data["Anxiety_Yes"]
y_test_panic = test_data["Panic Attack_Yes"]

# Scale features for Logistic Regression and KNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize results storage
evaluation_results = []

# Function to train and evaluate a model
def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, model_name, target):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    report = classification_report(y_test, y_pred, output_dict=True)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else "N/A"

    evaluation_results.append({
        "Model": model_name,
        "Target": target,
        "Accuracy": accuracy,
        "Precision (Macro)": report["macro avg"]["precision"],
        "Recall (Macro)": report["macro avg"]["recall"],
        "F1-Score (Macro)": report["macro avg"]["f1-score"],
        "Precision (Weighted)": report["weighted avg"]["precision"],
        "Recall (Weighted)": report["weighted avg"]["recall"],
        "F1-Score (Weighted)": report["weighted avg"]["f1-score"],
        "ROC AUC": roc_auc
    })

# Evaluate models for each target
targets = {
    "Depression": (y_train_depression, y_test_depression),
    "Anxiety": (y_train_anxiety, y_test_anxiety),
    "Panic Attack": (y_train_panic, y_test_panic)
}

models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

for target, (y_train, y_test) in targets.items():
    for model_name, model in models.items():
        X_train_data = X_train_scaled if model_name != "Decision Tree" else X_train
        X_test_data = X_test_scaled if model_name != "Decision Tree" else X_test
        train_and_evaluate_model(model, X_train_data, X_test_data, y_train, y_test, model_name, target)

# Save results to Excel
results_df = pd.DataFrame(evaluation_results)
output_file = "Evaluation_Results_Updated.xlsx"
results_df.to_excel(output_file, index=False)
print(f"Evaluation results saved to {output_file}.")