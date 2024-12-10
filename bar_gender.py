import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel("Student_Mental_Health_Cleaned.xlsx")

issues_of_interest = ['Depression', 'Anxiety', 'Panic Attack']

reshaped_data = pd.melt(df, id_vars='Gender', value_vars=issues_of_interest,
                        var_name='Mental_Health_Issue', value_name='Status')

filtered_data = reshaped_data.loc[reshaped_data['Status'] == 'Yes']

plt.figure(figsize=(8, 5))
sns.countplot(data=filtered_data, x='Mental_Health_Issue', hue='Gender')

# results
plt.title("What proportion of mental health problem based on gender ?")
plt.xlabel("mental health problem")
plt.ylabel("number of people")
plt.show()


