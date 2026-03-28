
# Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Advanced styling
plt.style.use('seaborn-v0_8-darkgrid')


# Load Dataset

df = pd.read_csv("zomato_sample_dataset.csv")


# Data Cleaning

df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# Convert rating to numeric (if string format)
df['rate'] = pd.to_numeric(df['rate'], errors='coerce')

# Convert cost column
df['approx_cost(for two people)'] = pd.to_numeric(
    df['approx_cost(for two people)'], errors='coerce'
)


# Feature Engineering

df['cost_per_person'] = df['approx_cost(for two people)'] / 2

# Split cuisines
df['cuisines'] = df['cuisines'].str.split(',')


# Top Cuisines Analysis

cuisines = df.explode('cuisines')
top_cuisines = cuisines['cuisines'].value_counts().head(10)

plt.figure(figsize=(10,6))
bars = plt.barh(top_cuisines.index, top_cuisines.values)

# Color grading
colors = plt.cm.viridis(np.linspace(0,1,len(bars)))
for bar, color in zip(bars, colors):
    bar.set_color(color)

plt.title("Top 10 Cuisines", fontsize=16, fontweight='bold')
plt.xlabel("Count")
plt.ylabel("Cuisine")
plt.gca().invert_yaxis()
plt.show()


# Rating Distribution

plt.figure(figsize=(8,5))

n, bins, patches = plt.hist(df['rate'], bins=20)

# Color gradient histogram
for i, patch in enumerate(patches):
    patch.set_facecolor(plt.cm.plasma(i/len(patches)))

plt.title("Rating Distribution", fontsize=15, fontweight='bold')
plt.xlabel("Ratings")
plt.ylabel("Frequency")
plt.show()


# Cost vs Rating Scatter (Advanced)

plt.figure(figsize=(8,6))

scatter = plt.scatter(
    df['approx_cost(for two people)'],
    df['rate'],
    c=df['rate'],
    cmap='coolwarm',
    alpha=0.7
)

plt.colorbar(scatter, label='Rating')
plt.title("Cost vs Rating", fontsize=15, fontweight='bold')
plt.xlabel("Cost for Two")
plt.ylabel("Rating")
plt.show()


# Online Order Impact

online = df.groupby('online_order')['rate'].mean()

plt.figure(figsize=(6,4))
bars = plt.bar(online.index, online.values)

# Custom colors
bars[0].set_color('#FF6F61')
bars[1].set_color('#6B5B95')

plt.title("Online Order vs Rating", fontsize=14, fontweight='bold')
plt.ylabel("Average Rating")
plt.show()


# Location Analysis

location = df['location'].value_counts().head(10)

plt.figure(figsize=(10,5))
bars = plt.bar(location.index, location.values)

# Smooth gradient
colors = plt.cm.inferno(np.linspace(0,1,len(bars)))
for bar, color in zip(bars, colors):
    bar.set_color(color)

plt.xticks(rotation=45)
plt.title("Top Locations", fontsize=15, fontweight='bold')
plt.xlabel("Location")
plt.ylabel("Number of Restaurants")
plt.show()


# Correlation Heatmap (Manual - Matplotlib)

corr = df[['rate','approx_cost(for two people)','cost_per_person']].corr()

plt.figure(figsize=(6,5))
plt.imshow(corr, cmap='coolwarm', interpolation='none')
plt.colorbar()

plt.xticks(range(len(corr)), corr.columns, rotation=45)
plt.yticks(range(len(corr)), corr.columns)

for i in range(len(corr)):
    for j in range(len(corr)):
        plt.text(j, i, f"{corr.iloc[i, j]:.2f}",
                 ha='center', va='center', color='black')

plt.title("Correlation Heatmap", fontweight='bold')
plt.show()


# BONUS: Top Restaurants

top_restaurants = df.sort_values(by='rate', ascending=False).head(10)
print(top_restaurants[['restaurant_name','rate','location']])