import pandas as pd
from tabulate import tabulate


file_path = '../../../Downloads/Student_Mental_Health_Cleaned.xlsx'
data = pd.read_excel(file_path)


if 'Timestamp' in data.columns:
    data = data.drop(columns=['Timestamp'])


numerical_cols = data.select_dtypes(include=['number']).columns
categorical_cols = data.select_dtypes(include=['object']).columns


if not numerical_cols.empty:
    print("\nSummary statistics for numerical columns:")

    summary_stats = data[numerical_cols].describe()

    summary_stats = summary_stats.drop(['25%', '50%', '75%'])

    summary_stats.rename(index={'std': 'Standard Deviation'}, inplace=True)
    print(tabulate(summary_stats, headers='keys', tablefmt='grid'))


if not categorical_cols.empty:
    print("\nSummary statistics for categorical columns:")
    for col in categorical_cols:
        print(f"\nColumn: {col}")

        value_counts = data[col].value_counts()

        print(tabulate(value_counts.reset_index(), headers=[col, 'Count'], tablefmt='grid'))