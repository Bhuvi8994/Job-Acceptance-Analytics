import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

st.title("Job Placement Prediction App")

# -------------------------------
# Step 1: Load Dataset
# -------------------------------
df = pd.read_csv(r"\Users\Bhuvaneswari\Downloads\Python\HR_Job_Placement_Dataset.csv")

df.columns = df.columns.str.strip().str.lower()
df = df.dropna()

# Convert target
df["status"] = df["status"].map({
    "Not Placed": 0,
    "Placed": 1
})

le = LabelEncoder()

for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

features = ['age_years',
            'ssc_percentage',
            'hsc_percentage',
       'degree_percentage',
       'certifications_count',
       'relevant_experience',
       'previous_ctc_lpa',
       'expected_ctc_lpa',
       'job_role_match',
       'competition_level',
       'notice_period_days',
       'employment_gap_months',
       'status', 
       'technical_score',
       'aptitude_score',
       'communication_score',
       'skills_match_percentage',
       'years_of_experience'
]

X = df[features]
y = df["status"]

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = LogisticRegression()
model.fit(X_scaled, y)

# -------------------------------
# Step 2: User Input
# -------------------------------
st.subheader("Enter Candidate Details")

age = st.slider("Age", 18, 60)
ssc = st.slider("SSC Percentage", 0, 100)
hsc = st.slider("HSC Percentage", 0, 100)
degree = st.slider("Degree Percentage", 0, 100)
cert = st.slider("Certifications Count", 0, 10)
relevant_exp = st.selectbox("Relevant Experience", [0, 1])
prev_ctc = st.number_input("Previous CTC (LPA)", 0.0, 50.0)
exp_ctc = st.number_input("Expected CTC (LPA)", 0.0, 50.0)
job_match = st.slider("Job Role Match %", 0, 100)
competition = st.selectbox("Competition Level", df['competition_level'].unique())
notice = st.slider("Notice Period (Days)", 0, 180)
gap = st.slider("Employment Gap (Months)", 0, 60)
tech = st.slider("Technical Score", 0, 100)
apt = st.slider("Aptitude Score", 0, 100)
comm = st.slider("Communication Score", 0, 100)
skills = st.slider("Skills Match %", 0, 100)
exp = st.slider("Years of Experience", 0, 10)

# -------------------------------
# Step 3: Prediction
# -------------------------------   
if st.button("Predict Placement"):

    input_data = np.array([[ 
        age,
        ssc,
        hsc,
        degree,
        cert,
        relevant_exp,
        prev_ctc,
        exp_ctc,
        job_match,
        competition,
        notice,
        gap,
        0,   
        tech,
        apt,
        comm,
        skills,
        exp
    ]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.success("🎉 Candidate will be Placed")
    else:
        st.error("⚠️ Candidate Not Placed")

    prob = model.predict_proba(input_scaled)[0][1]
    st.info(f"Placement Probability: {round(prob*100,2)} %")