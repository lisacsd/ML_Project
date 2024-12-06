import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_excel("./Preprocessed_Student_Mental_Health.xlsx")


mental_pb = ['Depression_Yes', 'Anxiety_Yes', 'Panic Attack_Yes']


pb_count = {pb: data[pb].sum() for pb in mental_pb}


labels = list(pb_count.keys())
sizes = list(pb_count.values())


plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title("Which mental health issue is present the most?")
plt.show()