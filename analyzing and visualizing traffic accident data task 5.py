import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap

# Step 1: Load the Traffic Accident Data
df = pd.read_csv('/Users/zach/Documents/traffic_accidents.csv')

# Convert Date column to datetime with the correct format
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%y', dayfirst=True)

# Convert Time column to datetime objects
df['Time'] = pd.to_datetime(df['Time'], format='%H:%M').dt.time

# Handle missing values by dropping rows with missing 'Latitude', 'Longitude', 'Weather_Conditions', or 'Road_Surface_Conditions'
df = df.dropna(subset=['Latitude', 'Longitude', 'Weather_Conditions', 'Road_Surface_Conditions'])

# Extract hour from the Time column for time-of-day analysis
df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.hour

# Display the cleaned data structure
print(df.info())

# Step 3: Exploratory Data Analysis (EDA)
# Accident Frequency by Hour of the Day
plt.figure(figsize=(10, 6))
sns.countplot(x='Hour', data=df, palette='viridis')
plt.title('Accident Frequency by Hour of the Day')
plt.xlabel('Hour of the Day')
plt.ylabel('Number of Accidents')
plt.show()

# Accident Frequency by Weather Condition
plt.figure(figsize=(10, 6))
sns.countplot(y='Weather_Conditions', data=df, palette='coolwarm', order=df['Weather_Conditions'].value_counts().index)
plt.title('Accident Frequency by Weather Condition')
plt.xlabel('Number of Accidents')
plt.ylabel('Weather Condition')
plt.show()

# Accident Frequency by Road Condition
plt.figure(figsize=(10, 6))
sns.countplot(y='Road_Surface_Conditions', data=df, palette='inferno', order=df['Road_Surface_Conditions'].value_counts().index)
plt.title('Accident Frequency by Road Condition')
plt.xlabel('Number of Accidents')
plt.ylabel('Road Surface Condition')
plt.show()

# Step 4: Visualize Accident Hotspots on a Map
# Create a base map centered around the mean location
m = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=12)

# Create a list of locations for the heatmap
heat_data = [[row['Latitude'], row['Longitude']] for index, row in df.iterrows()]

# Add the heatmap to the map
HeatMap(heat_data).add_to(m)

# Save the map as an HTML file
m.save("accident_hotspots.html")

# Step 5: Explore the Relationship Between Weather, Road Conditions, and Accidents
# Cross-tabulation of Weather and Road Conditions
cross_tab = pd.crosstab(df['Weather_Conditions'], df['Road_Surface_Conditions'])

# Plot a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(cross_tab, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Accident Frequency: Weather vs. Road Conditions')
plt.xlabel('Road Surface Condition')
plt.ylabel('Weather Condition')
plt.show()