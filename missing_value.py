import pandas as pd
from tabulate import tabulate

# Load the data
data = pd.read_excel("./Student_Mental_Health_Cleaned (1).xlsx")

# Missing values
missing_details = []
for column in data.columns:
    missing_rows = data[data[column].isnull()].index.tolist()
    if len(missing_rows) > 0:
        missing_details.append({
            'Column': column,
            'Line': ', '.join(map(lambda x: str(x + 2), missing_rows))  # Adjust for Excel line numbering
        })

missing_frame = pd.DataFrame(missing_details)

# Outliers
outlier_details = []
numeric_cols = data.select_dtypes(include=['number']).columns
for column in numeric_cols:
    if data[column].notnull().any():
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_limit = Q1 - 1.5 * IQR
        upper_limit = Q3 + 1.5 * IQR

        outlier_rows = data[(data[column] < lower_limit) | (data[column] > upper_limit)].index.tolist()
        if len(outlier_rows) > 0:
            outlier_details.append({
                'Column': column,
                'Rows': ', '.join(map(str, outlier_rows))
            })

outlier_frame = pd.DataFrame(outlier_details)

# Print results
print("\nMissing Data:")
if not missing_frame.empty:
    print(tabulate(missing_frame, headers='keys', tablefmt='grid'))
else:
    print("No missing data")

print("\nOutlier Info:")
if not outlier_frame.empty:
    print(tabulate(outlier_frame, headers='keys', tablefmt='grid'))
else:
    print("No outliers found")