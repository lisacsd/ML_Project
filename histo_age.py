import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel("./Student_Mental_Health_Cleaned.xlsx")

plt.figure(figsize=(8, 6))
plt.hist(df['Age'], bins=10, color='lightblue', edgecolor='gray')
plt.title("the age distribution")
plt.xlabel("age")
plt.ylabel("number of students")
plt.show()



