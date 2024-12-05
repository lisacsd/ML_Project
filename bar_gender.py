import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_excel("./Student_Mental_Health_Cleaned (1).xlsx")

mental_pb = ['Depression', 'Anxiety', 'Panic Attack']
data_long = data.melt(id_vars='Gender', value_vars=mental_pb,
                      var_name='Mental Problem', value_name='Presence')

data_long = data_long[data_long['Presence'] == 'Yes']

plt.figure(figsize=(10, 6))
sns.countplot(data=data_long, x='Mental Problem', hue='Gender')
plt.title("What proportion of mental health problem based on gender ?")
plt.xlabel("mental health problem")
plt.ylabel("number of people")
plt.show()