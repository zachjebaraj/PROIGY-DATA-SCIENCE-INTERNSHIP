import matplotlib.pyplot as plt

# Age distribution data
age_groups = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80']
population = [5, 8, 12, 7, 4, 6, 3, 2]

# Create the bar chart
plt.figure(figsize=(10, 6))
plt.bar(age_groups, population, color='skyblue')
plt.xlabel('Age Groups')
plt.ylabel('Number of People')
plt.title('Age Distribution in a Population')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
