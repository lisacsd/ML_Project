import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel("Student_Mental_Health_Cleaned.xlsx")

issues = ['Depression', 'Anxiety', 'Panic Attack']
values = [df[issue].value_counts().get('Yes', 0) for issue in issues]

plt.figure(figsize=(7, 7))
plt.pie(values, labels=issues, autopct='%1.1f%%', startangle=140)
plt.title("Which mental health is present the most?")
plt.show()
