import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


file_path = 'Student_Mental_Health_Cleaned.xlsx'

data = pd.read_excel(file_path)


data = data.drop(columns=['Timestamp'], errors='ignore')


categorical_features = ['Gender', 'Course', 'Year of Study', 'CGPA',
                        'Marital Status', 'Depression', 'Anxiety', 'Panic Attack', 'Specialist Treatment']


data[categorical_features] = data[categorical_features].fillna("Unknown")


numerical_features = ['Age']


preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
    ],
    remainder='passthrough'
)

