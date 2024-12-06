import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

file_path = 'Student_Mental_Health_Cleaned.xlsx'
data = pd.read_excel(file_path)

data = data.drop(columns=['Timestamp'])

categorical_features = ['Gender', 'Course', 'Year of Study', 'CGPA',
                        'Marital Status', 'Depression', 'Anxiety', 'Panic Attack', 'Specialist Treatment']

data[categorical_features] = data[categorical_features].fillna("Unknown")

data['Age_Unchanged'] = data['Age']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
    ]
)

preprocessed_data = preprocessor.fit_transform(data)


cat_col_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)


preprocessed_df = pd.DataFrame(preprocessed_data, columns=cat_col_names)

preprocessed_df.insert(0, 'Age', data['Age_Unchanged'].reset_index(drop=True))


output_file = "Preprocessed_Student_Mental_Health.xlsx"
preprocessed_df.to_excel(output_file, index=False)

print(f"File saved: {output_file}")
