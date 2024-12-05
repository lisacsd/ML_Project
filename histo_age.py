import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_excel("./Student_Mental_Health_Cleaned (1).xlsx")

plt.figure(figsize=(8, 6))
plt.hist(data['Age'], bins=10, color='skyblue', edgecolor='black')
plt.title("the age distribution")
plt.xlabel("age")
plt.ylabel("number of students")
plt.show()