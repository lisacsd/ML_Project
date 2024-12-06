import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

file_path = 'Student_Mental_Health_Cleaned (1).xlsx'
data = pd.read_excel(file_path)

# Drop unnecessary columns
data = data.drop(columns=['Timestamp'])

# Define numerical and categorical features
numerical_features = ['Age']
categorical_features = ['Gender', 'Course', 'Year of Study', 'CGPA',
                        'Marital Status', 'Depression', 'Anxiety', 'Panic Attack', 'Specialist Treatment']

# Fill missing values
data[numerical_features] = data[numerical_features].fillna(data[numerical_features].mean())
data[categorical_features] = data[categorical_features].fillna("Unknown")

# Define the preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
    ])

# Fit and transform the data
preprocessed_data = preprocessor.fit_transform(data)

# Get column names for the preprocessed data
num_col_names = numerical_features
cat_col_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
all_col_names = list(num_col_names) + list(cat_col_names)

# Create a DataFrame for the preprocessed data
preprocessed_df = pd.DataFrame(preprocessed_data, columns=all_col_names)

# Save the DataFrame as an Excel file
output_file = "Preprocessed_Student_Mental_Health.xlsx"
preprocessed_df.to_excel(output_file, index=False)

print(f"File saved: {output_file}")