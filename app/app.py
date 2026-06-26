import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load artifacts
model = pickle.load(open("models/model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))
features = pickle.load(open("models/features.pkl", "rb"))

st.set_page_config(page_title="Student Performance Predictor", layout="centered")

st.title("🎓 Student Performance Predictor")
st.write("Enter student details to predict exam score")

# -----------------------------
# Numeric Inputs
# -----------------------------
hours_studied = st.slider("Hours Studied", 0, 12, 6)
attendance = st.slider("Attendance", 0, 100, 75)
sleep_hours = st.slider("Sleep Hours", 0, 12, 7)
previous_scores = st.slider("Previous Scores", 0, 100, 70)
tutoring_sessions = st.slider("Tutoring Sessions", 0, 10, 2)
physical_activity = st.slider("Physical Activity", 0, 10, 5)

# -----------------------------
# Categorical Inputs
# -----------------------------
parental_involvement = st.selectbox("Parental Involvement", ["Low", "Medium", "High"])
access_resources = st.selectbox("Access to Resources", ["Low", "Medium", "High"])
extracurricular = st.selectbox("Extracurricular Activities", ["Yes", "No"])
motivation = st.selectbox("Motivation Level", ["Low", "Medium", "High"])
internet = st.selectbox("Internet Access", ["Yes", "No"])
family_income = st.selectbox("Family Income", ["Low", "Medium", "High"])
teacher_quality = st.selectbox("Teacher Quality", ["Low", "Medium", "High", "Unknown"])
school_type = st.selectbox("School Type", ["Public", "Private"])
peer_influence = st.selectbox("Peer Influence", ["Negative", "Neutral", "Positive"])
learning_disabilities = st.selectbox("Learning Disabilities", ["Yes", "No"])
parent_edu = st.selectbox("Parental Education Level", ["High School", "Graduate", "Postgraduate", "Unknown"])
distance = st.selectbox("Distance from Home", ["Near", "Moderate", "Far"])
gender = st.selectbox("Gender", ["Male", "Female"])

# -----------------------------
# Build Input Data
# -----------------------------
input_data = {
    "Hours_Studied": hours_studied,
    "Attendance": attendance,
    "Sleep_Hours": sleep_hours,
    "Previous_Scores": previous_scores,
    "Tutoring_Sessions": tutoring_sessions,
    "Physical_Activity": physical_activity,

    "Parental_Involvement": parental_involvement,
    "Access_to_Resources": access_resources,
    "Extracurricular_Activities": extracurricular,
    "Motivation_Level": motivation,
    "Internet_Access": internet,
    "Family_Income": family_income,
    "Teacher_Quality": teacher_quality,
    "School_Type": school_type,
    "Peer_Influence": peer_influence,
    "Learning_Disabilities": learning_disabilities,
    "Parental_Education_Level": parent_edu,
    "Distance_from_Home": distance,
    "Gender": gender
}

input_df = pd.DataFrame([input_data])

# Encode like training
input_df = pd.get_dummies(input_df)

# Align with training features
input_df = input_df.reindex(columns=features, fill_value=0)

# Scale
input_scaled = scaler.transform(input_df)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Score"):

    prediction = model.predict(input_scaled)

    st.success(f"🎯 Predicted Exam Score: {prediction[0]:.2f}")

    # -----------------------------
    # Insight Section
    # -----------------------------
    st.subheader("📊 Prediction Insight")

    if hasattr(model, "coef_"):
        importance_df = pd.DataFrame({
            "Feature": features,
            "Impact": model.coef_
        })

        importance_df = importance_df.sort_values(by="Impact", ascending=False).head(10)

        st.write("Top contributing factors (based on model coefficients):")
        st.dataframe(importance_df)

        st.write(f"Most influential feature: **{importance_df.iloc[0]['Feature']}**")