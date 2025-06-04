import streamlit as st
import pandas as pd
import joblib

st.title("Prediksi Stroke")

model = joblib.load("models/stroke_model.pkl")

def user_input_features():
    gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
    age = st.slider("Usia", 0, 100, 25)
    hypertension = st.selectbox("Hipertensi", [0, 1])
    heart_disease = st.selectbox("Penyakit Jantung", [0, 1])
    ever_married = st.selectbox("Pernah Menikah", ["Yes", "No"])
    work_type = st.selectbox("Jenis Pekerjaan", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    Residence_type = st.selectbox("Tipe Tempat Tinggal", ["Urban", "Rural"])
    avg_glucose_level = st.slider("Rata-rata Glukosa", 50.0, 300.0, 100.0)
    bmi = st.slider("BMI", 10.0, 50.0, 20.0)
    smoking_status = st.selectbox("Status Merokok", ["formerly smoked", "never smoked", "smokes", "Unknown"])

    data = {
        'gender': gender,
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'ever_married': ever_married,
        'work_type': work_type,
        'Residence_type': Residence_type,
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,
        'smoking_status': smoking_status
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()
st.subheader("Data Input")
st.write(input_df)

# Preprocessing sesuai dengan pelatihan model
df = pd.read_csv("healthcare-dataset-stroke-data.csv")
df = df.dropna()
df = pd.get_dummies(df, drop_first=True)
input_df = pd.get_dummies(input_df)
input_df = input_df.reindex(columns=df.drop("stroke", axis=1).columns, fill_value=0)

prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.subheader("Hasil Prediksi")
st.write("Kemungkinan Stroke: ", "Ya" if prediction[0] == 1 else "Tidak")
st.write("Probabilitas: ", prediction_proba[0][prediction[0]])
