import pandas as pd

file_path = 'Student_Mental_Health_Cleaned.xlsx'
data = pd.read_excel(file_path)


if 'Timestamp' in data.columns:
    data.drop(columns=['Timestamp'], inplace=True)


output_file = "Summary_Statistics.xlsx"
writer = pd.ExcelWriter(output_file, engine='xlsxwriter')
sheet = writer.book.add_worksheet("Summary")


col_num = 0


age_summary = data['Age'].describe().drop(['25%', '50%', '75%']).rename({'std': 'Standard Deviation'}).reset_index()
age_summary.columns = ['Age', 'Value']
age_summary.to_excel(writer, sheet_name='Summary', index=False, startrow=0, startcol=col_num)
col_num += 3


categorical_cols = data.select_dtypes(include=['object']).columns


for col in categorical_cols:
    sheet.write(0, col_num, col)


    value_counts = data[col].value_counts().reset_index()
    value_counts.columns = ['Value', 'Count']


    value_counts.to_excel(writer, sheet_name='Summary', index=False, startrow=1, startcol=col_num)
    col_num += 3


writer.close()
print(f"Summary saved in {output_file}")