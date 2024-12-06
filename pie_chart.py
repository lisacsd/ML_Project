import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_excel("./Student_Mental_Health_Cleaned.xlsx")


mental_pb = ['Depression', 'Anxiety', 'Panic Attack']
pb_count = {pb: data[pb].value_counts()['Yes'] for pb in mental_pb}


labels = list(pb_count.keys())
sizes = list(pb_count.values())


plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title("Which mental health is present the most?")
plt.show()
