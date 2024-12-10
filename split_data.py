import pandas as pd
from sklearn.model_selection import train_test_split

input_file = "Preprocessed_Student_Mental_Health.xlsx"
df = pd.read_excel(input_file)

features = df.drop(columns=["Depression_Yes", "Anxiety_Yes", "Panic Attack_Yes"])
target_dep = df["Depression_Yes"]
target_anx = df["Anxiety_Yes"]
target_pan = df["Panic Attack_Yes"]

X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(features, target_dep, test_size=0.2, random_state=42)
X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(features, target_anx, test_size=0.2, random_state=42)
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(features, target_pan, test_size=0.2, random_state=42)

train_set = pd.concat([X_train_d, y_train_d, y_train_a, y_train_p], axis=1)
test_set = pd.concat([X_test_d, y_test_d, y_test_a, y_test_p], axis=1)

train_file = "Train_Student_Mental_Health.xlsx"
train_set.to_excel(train_file, index=False)

test_file = "Test_Student_Mental_Health.xlsx"
test_set.to_excel(test_file, index=False)

print(f"Saved training dataset to {train_file}")
print(f"Saved testing dataset to {test_file}")