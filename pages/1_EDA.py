import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Exploratory Data Analysis (EDA)")

@st.cache_data
def load_data():
    return pd.read_csv("data/healthcare-dataset-stroke-data.csv")

df = load_data()
st.write("### Dataset")
st.dataframe(df.head())

st.write("### Informasi Umum")
st.write(df.describe())

st.write("### Distribusi Stroke")
fig, ax = plt.subplots()
sns.countplot(x='stroke', data=df, ax=ax)
st.pyplot(fig)

