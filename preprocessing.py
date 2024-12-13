import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

path = 'Student_Mental_Health_Cleaned.xlsx'
df = pd.read_excel(path)
df = df.drop(columns=['Timestamp'], errors='ignore')

categories = ['Gender', 'Course', 'Year of Study', 'CGPA',
              'Marital Status', 'Depression', 'Anxiety', 'Panic Attack', 'Specialist Treatment']

df[categories] = df[categories].fillna("Unknown")

numerics = ['Age']

transformer = ColumnTransformer(
    transformers=[
        ('encoder', OneHotEncoder(drop='first', sparse_output=False), categories)
    ],
    remainder='passthrough'
)

transformed_data = transformer.fit_transform(df)
encoded_columns = transformer.named_transformers_['encoder'].get_feature_names_out(categories)
all_columns = list(encoded_columns) + numerics
final_df = pd.DataFrame(transformed_data, columns=all_columns)
ordered_columns = ['Age'] + [col for col in final_df.columns if col != 'Age']
final_df = final_df[ordered_columns]

output = "Preprocessed_Student_Mental_Health.xlsx"
final_df.to_excel(output, index=False)

print(f"Saved file: {output}")