import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# Load preprocessed training and testing data
train_df = pd.read_excel("Train_Student_Mental_Health.xlsx")
test_df = pd.read_excel("Test_Student_Mental_Health.xlsx")

# Define features and targets for training and testing
X_train = train_df.drop(['Depression_Yes', 'Anxiety_Yes', 'Panic Attack_Yes'], axis=1)
y_train = {
    "Depression": train_df['Depression_Yes'],
    "Anxiety": train_df['Anxiety_Yes'],
    "Panic Attack": train_df['Panic Attack_Yes']
}

X_test = test_df.drop(['Depression_Yes', 'Anxiety_Yes', 'Panic Attack_Yes'], axis=1)
y_test = {
    "Depression": test_df['Depression_Yes'],
    "Anxiety": test_df['Anxiety_Yes'],
    "Panic Attack": test_df['Panic Attack_Yes']
}

# Initialize models and hyperparameters
classifiers = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "KNN": KNeighborsClassifier()
}

parameters = {
    "Decision Tree": {'max_depth': [3, 5, 10, None]},
    "Logistic Regression": {'C': [0.1, 1, 10, 100]},
    "KNN": {'n_neighbors': [3, 5, 7, 9], 'metric': ['euclidean', 'manhattan']}
}

# Store evaluation results
evaluation_results = []

# Model evaluation process
for clf_name, clf in classifiers.items():
    for target in y_train.keys():
        grid_search = GridSearchCV(clf, parameters[clf_name], cv=5)
        grid_search.fit(X_train, y_train[target])

        preds = grid_search.predict(X_test)
        probs = grid_search.predict_proba(X_test)[:, 1]

        report = classification_report(y_test[target], preds, output_dict=True)
        roc_auc = roc_auc_score(y_test[target], probs)

        evaluation_results.append({
            "Model": clf_name,
            "Target": target,
            "Precision": round(report['weighted avg']['precision'], 3),
            "Recall": round(report['weighted avg']['recall'], 3),
            "F1-Score": round(report['weighted avg']['f1-score'], 3),
            "ROC AUC": round(roc_auc, 3)
        })

# Save results to an Excel file
output_file_name = "Model_Evaluation_Summary_Train_Test.xlsx"
results_df = pd.DataFrame(evaluation_results)
results_df.to_excel(output_file_name, index=False)

print(f"Results saved to {output_file_name}")