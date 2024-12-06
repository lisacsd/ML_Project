import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tkinter import Tk, Frame, Scrollbar, ttk

# Load the dataset
df = pd.read_excel("./Student_Mental_Health_Cleaned (1).xlsx")

# 1. Handle Missing Values
print("=== Handling Missing Values ===")
# Replace missing values in numerical columns with the mean
numeric_cols_with_na = ['Age']  # Update this list with numerical columns that may have missing values
for col in numeric_cols_with_na:
    mean_value = df[col].mean()
    print(f"Replacing missing values in '{col}' with mean: {mean_value:.2f}")
    df[col] = df[col].fillna(mean_value)

# Replace missing values in categorical columns with 'Unknown'
categorical_cols_with_na = ['Specialist Treatment']  # Update this list with your categorical columns
for col in categorical_cols_with_na:
    print(f"Replacing missing values in '{col}' with: 'Unknown'")
    df[col] = df[col].fillna('Unknown')

# 2. Normalize Numerical Columns
print("\n=== Normalizing Numerical Columns ===")
scaler = MinMaxScaler()
numeric_cols = ['Age']  # Update this list with all numerical columns in your dataset
print("Before normalization:")
print(df[numeric_cols].head())
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
print("After normalization:")
print(df[numeric_cols].head())

# 3. Encode Categorical Columns
print("\n=== Encoding Categorical Columns ===")
categorical_cols = ['Specialist Treatment']  # Update this list with all categorical columns in your dataset
label_encoder = LabelEncoder()
for col in categorical_cols:
    print(f"Encoding '{col}': {df[col].unique()} -> {list(label_encoder.fit(df[col]).classes_)}")
    df[col] = label_encoder.fit_transform(df[col])

# Function to display DataFrame in a Tkinter window
def show_dataframe(dataframe, title):
    root = Tk()
    root.title(title)
    frame = Frame(root)
    frame.pack(fill="both", expand=True)

    # Scrollbars
    vsb = Scrollbar(frame, orient="vertical")
    vsb.pack(side="right", fill="y")
    hsb = Scrollbar(frame, orient="horizontal")
    hsb.pack(side="bottom", fill="x")

    # Table Treeview
    tree = ttk.Treeview(frame, columns=list(dataframe.columns), show="headings", yscrollcommand=vsb.set, xscrollcommand=hsb.set)
    vsb.config(command=tree.yview)
    hsb.config(command=tree.xview)

    # Add columns to the Treeview
    for col in dataframe.columns:
        tree.heading(col, text=col)
        tree.column(col, anchor="center", width=100)

    # Add rows to the Treeview
    for _, row in dataframe.iterrows():
        tree.insert("", "end", values=list(row))

    tree.pack(expand=True, fill="both")
    root.mainloop()

# Display DataFrames in Tkinter
show_dataframe(df.head(10), "Top 10 Rows")  # Display the first 10 rows
show_dataframe(df.describe().transpose(), "Summary Statistics")  # Display summary statistics
show_dataframe(df.isnull().sum().reset_index(name="Missing Values"), "Missing Values")  # Display missing values summary

# Display encoded categorical columns
categorical_data = df[categorical_cols].head(10)
show_dataframe(categorical_data, "Encoded Categorical Columns")