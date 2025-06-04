import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

st.title("Pelatihan Model")

@st.cache_data
def load_data():
    return pd.read_csv("data/healthcare-dataset-stroke-data.csv")

df = load_data()
df = df.dropna()
df = pd.get_dummies(df, drop_first=True)

X = df.drop("stroke", axis=1)
y = df["stroke"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

st.write("### Hasil Evaluasi Model")
st.text(classification_report(y_test, y_pred))

joblib.dump(model, "models/stroke_model.pkl")
st.success("Model telah disimpan di folder 'models'.")
