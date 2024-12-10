import pandas as pd

input_file = 'Student_Mental_Health_Cleaned.xlsx'
df = pd.read_excel(input_file)

if 'Timestamp' in df.columns:
    df.drop(columns=['Timestamp'], inplace=True)

output_summary = "Mental_Health_Summary.xlsx"
excel_writer = pd.ExcelWriter(output_summary, engine='xlsxwriter')
worksheet = excel_writer.book.add_worksheet("Overview")

column_index = 0

age_stats = df['Age'].describe().drop(['25%', '50%', '75%']).rename({'std': 'Std Dev'}).reset_index()
age_stats.columns = ['Age Stats', 'Values']
age_stats.to_excel(excel_writer, sheet_name='Overview', index=False, startrow=0, startcol=column_index)
column_index += 3

category_columns = df.select_dtypes(include=['object']).columns

for category in category_columns:
    worksheet.write(0, column_index, category)

    category_counts = df[category].value_counts().reset_index()
    category_counts.columns = ['Category Value', 'Frequency']

    category_counts.to_excel(excel_writer, sheet_name='Overview', index=False, startrow=1, startcol=column_index)
    column_index += 3

excel_writer.close()
print(f"File saved: {output_summary}")