
# Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Premium styles
plt.style.use('seaborn-v0_8-darkgrid')


# Load Dataset

df = pd.read_csv("credit_dataset.csv")


# Data Cleaning

df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

df['Date'] = pd.to_datetime(df['Date'])
df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')


# Feature Engineering

df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day_name()
df['YearMonth'] = df['Date'].dt.to_period('M')


# Monthly Spending Trend (Line Plot)

monthly = df.groupby('YearMonth')['Amount'].sum()

plt.figure(figsize=(12,5))
plt.plot(monthly.index.astype(str), monthly.values, marker='o', linewidth=2)

# Gradient color line
for i in range(len(monthly)-1):
    plt.plot(monthly.index.astype(str)[i:i+2],
             monthly.values[i:i+2],
             color=plt.cm.viridis(i/len(monthly)))

plt.title("Monthly Spending Trend", fontsize=16, fontweight='bold')
plt.xticks(rotation=45)
plt.xlabel("Month")
plt.ylabel("Spending")
plt.show()


# Category-wise Spending (Bar Chart)

category = df.groupby('Category')['Amount'].sum().sort_values(ascending=False)

plt.figure(figsize=(10,5))
bars = plt.bar(category.index, category.values)

colors = plt.cm.plasma(np.linspace(0,1,len(bars)))
for bar, color in zip(bars, colors):
    bar.set_color(color)

plt.title("Spending by Category", fontsize=15, fontweight='bold')
plt.xticks(rotation=45)
plt.show()


# Spending Distribution (Histogram)

plt.figure(figsize=(8,5))
n, bins, patches = plt.hist(df['Amount'], bins=30)

for i, patch in enumerate(patches):
    patch.set_facecolor(plt.cm.inferno(i/len(patches)))

plt.title("Transaction Distribution", fontsize=14, fontweight='bold')
plt.xlabel("Amount")
plt.ylabel("Frequency")
plt.show()


# Day-wise Spending (Bar Chart)

day_spending = df.groupby('Day')['Amount'].sum()

plt.figure(figsize=(8,5))
bars = plt.bar(day_spending.index, day_spending.values)

colors = plt.cm.coolwarm(np.linspace(0,1,len(bars)))
for bar, color in zip(bars, colors):
    bar.set_color(color)

plt.title("Spending by Day", fontsize=14, fontweight='bold')
plt.xticks(rotation=45)
plt.show()


# Cost vs Frequency Scatter Plot

plt.figure(figsize=(8,6))

scatter = plt.scatter(
    df['Month'],
    df['Amount'],
    c=df['Amount'],
    cmap='viridis',
    alpha=0.6
)

plt.colorbar(scatter, label="Transaction Value")
plt.title("Spending Pattern Heat Scatter", fontsize=14, fontweight='bold')
plt.xlabel("Month")
plt.ylabel("Amount")
plt.show()


# Top Customers (Horizontal Bar)

top_customers = df.groupby('CustomerID')['Amount'].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(8,5))
bars = plt.barh(top_customers.index.astype(str), top_customers.values)

colors = plt.cm.magma(np.linspace(0,1,len(bars)))
for bar, color in zip(bars, colors):
    bar.set_color(color)

plt.gca().invert_yaxis()
plt.title("Top 10 Customers", fontsize=14, fontweight='bold')
plt.show()


# Correlation Heatmap 
corr = df[['Amount','Month']].corr()

plt.figure(figsize=(5,4))
plt.imshow(corr, cmap='coolwarm', interpolation='none')
plt.colorbar()

plt.xticks(range(len(corr)), corr.columns)
plt.yticks(range(len(corr)), corr.columns)

for i in range(len(corr)):
    for j in range(len(corr)):
        plt.text(j, i, f"{corr.iloc[i,j]:.2f}",
                 ha='center', va='center', color='black')

plt.title("Correlation Heatmap", fontweight='bold')
plt.show()


# Pie Chart (Category Share)

plt.figure(figsize=(6,6))
plt.pie(category.values, labels=category.index, autopct='%1.1f%%',
        colors=plt.cm.Set3(np.linspace(0,1,len(category))))

plt.title("Category Contribution", fontweight='bold')
plt.show()


# Customer Segmentation (RFM-like)

customer_spending = df.groupby('CustomerID')['Amount'].sum()

segments = pd.qcut(customer_spending, q=3, labels=['Low','Medium','High'])

segment_count = segments.value_counts()

plt.figure(figsize=(6,4))
bars = plt.bar(segment_count.index, segment_count.values)

colors = ['#4CAF50','#FFC107','#F44336']
for bar, color in zip(bars, colors):
    bar.set_color(color)

plt.title("Customer Segmentation", fontweight='bold')
plt.show()