import pandas as pd
from sklearn.model_selection import train_test_split

file_path = "Preprocessed_Student_Mental_Health.xlsx"
data = pd.read_excel(file_path)


X = data.drop(columns=["Depression_Yes", "Anxiety_Yes", "Panic Attack_Yes"])
y_depression = data["Depression_Yes"]
y_anxiety = data["Anxiety_Yes"]
y_panic = data["Panic Attack_Yes"]

# split (80% train, 20% test)
X_train_dep, X_test_dep, y_train_dep, y_test_dep = train_test_split(X, y_depression, test_size=0.2, random_state=42)
X_train_anx, X_test_anx, y_train_anx, y_test_anx = train_test_split(X, y_anxiety, test_size=0.2, random_state=42)
X_train_pan, X_test_pan, y_train_pan, y_test_pan = train_test_split(X, y_panic, test_size=0.2, random_state=42)


train_data = pd.concat([X_train_dep, y_train_dep, y_train_anx, y_train_pan], axis=1)
test_data = pd.concat([X_test_dep, y_test_dep, y_test_anx, y_test_pan], axis=1)


train_output_path = "./Train_Student_Mental_Health.xlsx"
train_data.to_excel(train_output_path, index=False)


test_output_path = "./Test_Student_Mental_Health.xlsx"
test_data.to_excel(test_output_path, index=False)

print(f"Training data has been saved to {train_output_path}")
print(f"Testing data has been saved to {test_output_path}")