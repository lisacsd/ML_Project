import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_excel("Student_Mental_Health_Cleaned.xlsx")


problems = ['Depression', 'Anxiety', 'Panic Attack']
data_melted = data.melt(id_vars='Gender', value_vars=problems,
                        var_name='Problem', value_name='Has_Issue')


data_melted = data_melted[data_melted['Has_Issue'] == 'Yes']

# results
plt.figure(figsize=(8, 5))
sns.countplot(data=data_melted, x='Problem', hue='Gender')
plt.title("What proportion of mental health problem based on gender ?")
plt.xlabel("mental health problem")
plt.ylabel("number of people")
plt.show()


