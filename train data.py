# Importing Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

# Load Data
data = pd.read_excel("Preprocessed_Student_Mental_Health.xlsx")

# Splitting Features and Targets
X = data.drop(['Depression_Yes', 'Anxiety_Yes', 'Panic Attack_Yes'], axis=1)
y_depression = data['Depression_Yes']
y_anxiety = data['Anxiety_Yes']
y_panic = data['Panic Attack_Yes']

# Train-Test Split
X_train, X_test, y_train_depression, y_test_depression = train_test_split(X, y_depression, test_size=0.2, random_state=42)
_, _, y_train_anxiety, y_test_anxiety = train_test_split(X, y_anxiety, test_size=0.2, random_state=42)
_, _, y_train_panic, y_test_panic = train_test_split(X, y_panic, test_size=0.2, random_state=42)

# Define Models
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "KNN": KNeighborsClassifier()
}

# Hyperparameter Tuning
model_params = {
    "Decision Tree": {'max_depth': [3, 5, 10, None]},
    "Logistic Regression": {'C': [0.1, 1, 10, 100]},
    "KNN": {'n_neighbors': [3, 5, 7, 9], 'metric': ['euclidean', 'manhattan']}
}

# Evaluation Function
def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    roc_score = roc_auc_score(y_test, y_proba)

    return {
        "Precision": report['weighted avg']['precision'],
        "Recall": report['weighted avg']['recall'],
        "F1-Score": report['weighted avg']['f1-score'],
        "ROC AUC": roc_score
    }

# Store Results
results = []

# Evaluate All Models
for model_name, model in models.items():
    for target_name, (y_train, y_test) in zip(
        ["Depression", "Anxiety", "Panic Attack"],
        [(y_train_depression, y_test_depression),
         (y_train_anxiety, y_test_anxiety),
         (y_train_panic, y_test_panic)]
    ):
        tuned_model = GridSearchCV(model, model_params[model_name], cv=5)
        scores = evaluate_model(tuned_model, X_train, y_train, X_test, y_test)
        results.append({
            "Model": model_name,
            "Target": target_name,
            "Precision": round(scores["Precision"], 3),
            "Recall": round(scores["Recall"], 3),
            "F1-Score": round(scores["F1-Score"], 3),
            "ROC AUC": round(scores["ROC AUC"], 3)
        })

# Convert to DataFrame and Save to Excel
results_df = pd.DataFrame(results)
output_file = "Model_Evaluation_Results.xlsx"
results_df.to_excel(output_file, index=False)

print(f"Results saved to {output_file}")