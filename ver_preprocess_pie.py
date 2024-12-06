
import pandas as pd
import matplotlib.pyplot as plt

# Load your data (ensure the file path is correct)
data = pd.read_excel("./Preprocessed_Student_Mental_Health.xlsx")  # Update with the correct file path or format

# Define the mental health problems of interest
mental_pb = ['Depression_Yes', 'Anxiety_Yes', 'Panic Attack_Yes']

# Count the occurrences of "Yes" for each mental health problem
pb_count = {pb: data[pb].sum() for pb in mental_pb}

# Extract labels and sizes for the pie chart
labels = list(pb_count.keys())
sizes = list(pb_count.values())

# Plot the pie chart
plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title("Which mental health issue is present the most?")
plt.show()