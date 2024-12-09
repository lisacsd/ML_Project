import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_excel("Student_Mental_Health_Cleaned.xlsx")


problems = ['Depression', 'Anxiety', 'Panic Attack']
counts = [data[pb].value_counts().get('Yes', 0) for pb in problems]

#pie chart
plt.figure(figsize=(7, 7))
plt.pie(counts, labels=problems, autopct='%1.1f%%', startangle=140)
plt.title("Which mental health is present the most?")
plt.show()
