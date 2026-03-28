import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Sales Data Analysis Dashboard")

# Upload file
file = st.file_uploader("Upload Sales CSV File", type=["csv"])

if file is not None:
    df = pd.read_csv(file)

    st.subheader("Dataset Preview")
    st.write(df)

    # Total Sales
    total_sales = df["Sales"].sum()
    st.metric("Total Sales", total_sales)

    # Sales by Product
    product_sales = df.groupby("Product")["Sales"].sum()

    st.subheader("Sales by Product (Bar Chart)")
    fig, ax = plt.subplots()
    product_sales.plot(kind="bar", ax=ax)
    st.pyplot(fig)

    # Sales by Category
    category_sales = df.groupby("Category")["Sales"].sum()

    st.subheader("Sales by Category (Pie Chart)")
    fig2, ax2 = plt.subplots()
    ax2.pie(category_sales, labels=category_sales.index, autopct="%1.1f%%")
    st.pyplot(fig2)

    # Donut Chart
    st.subheader("Sales Distribution (Donut Chart)")
    fig3, ax3 = plt.subplots()
    ax3.pie(category_sales, labels=category_sales.index, autopct="%1.1f%%")
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig3.gca().add_artist(centre_circle)
    st.pyplot(fig3)