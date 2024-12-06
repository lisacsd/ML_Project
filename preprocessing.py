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


preprocessed_data = preprocessor.fit_transform(data)


cat_col_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)


all_feature_names = list(cat_col_names) + numerical_features


preprocessed_df = pd.DataFrame(preprocessed_data, columns=all_feature_names)

columns_order = ['Age'] + [col for col in preprocessed_df.columns if col != 'Age']
preprocessed_df = preprocessed_df[columns_order]


output_file = "Preprocessed_Student_Mental_Health.xlsx"
preprocessed_df.to_excel(output_file, index=False)

print(f"File saved: {output_file}")