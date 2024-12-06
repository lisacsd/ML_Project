import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# File path
file_path = 'Student_Mental_Health_Cleaned.xlsx'

# Load the dataset
data = pd.read_excel(file_path)

# Drop irrelevant columns (if confirmed unnecessary)
data = data.drop(columns=['Timestamp'], errors='ignore')  # Avoid errors if column doesn't exist

# Specify categorical features
categorical_features = ['Gender', 'Course', 'Year of Study', 'CGPA',
                        'Marital Status', 'Depression', 'Anxiety', 'Panic Attack', 'Specialist Treatment']

# Fill missing values for categorical columns
data[categorical_features] = data[categorical_features].fillna("Unknown")

# Preserve numerical columns if needed
numerical_features = ['Age']  # Add other numerical columns if necessary

# One-Hot Encoding for categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
    ],
    remainder='passthrough'  # Keeps other columns (e.g., numerical data) intact
)

# Fit and transform data
preprocessed_data = preprocessor.fit_transform(data)

# Get feature names
cat_col_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)

# Combine numerical and categorical feature names
all_feature_names = list(cat_col_names) + numerical_features

# Create a DataFrame
preprocessed_df = pd.DataFrame(preprocessed_data, columns=all_feature_names)

# Save to Excel
output_file = "Preprocessed_Student_Mental_Health.xlsx"
preprocessed_df.to_excel(output_file, index=False)

print(f"File saved: {output_file}")