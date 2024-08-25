import pandas as pd

# Load the Titanic dataset (adjust the path to where your file is located)
titanic_df = pd.read_csv('/Users/zach/Downloads/Titanic-Dataset.csv')

# Display the first few rows to verify
print(titanic_df.head())

import seaborn as sns
import matplotlib.pyplot as plt

# Assuming titanic_df is your DataFrame containing the Titanic data

# Plot: Survival rate by Gender
plt.figure(figsize=(8, 6))
sns.countplot(data=titanic_df, x='Sex', hue='Survived', palette='muted')
plt.title('Survival Rate by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.legend(title='Survived', loc='upper right')
plt.show()