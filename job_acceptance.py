import streamlit as st
import pandas as pd

df = pd.read_csv(r"C:\Users\Bhuvaneswari\Downloads\Python\HR_Job_Placement_Dataset.csv")

st.title("Job Acceptance Analytics Dashboard")

df["status"] = df["status"].map({
    "Not Placed": 0,
    "Placed": 1
})

# Total candidates
total_candidates = len(df)

# Placement Rate
placement_rate = (df["status"].mean()) * 100

# Job Acceptance Rate
job_acceptance_rate = placement_rate

# Interview Score
df["interview_score"] = (
    df["technical_score"] +
    df["aptitude_score"] +
    df["communication_score"]
) 

avg_interview_score = df["interview_score"].mean()

# Average Skills Match
avg_skills_match = df["skills_match_percentage"].mean()

# Offer Dropout Rate
offer_dropout_rate = (df["status"] == 0).mean() * 100

# High Risk Candidates (skills < 50%)
high_risk_candidates = (
    df["skills_match_percentage"] < 50
).mean() * 100



col1, col2, col3 = st.columns(3)

col1.metric("Total Candidates", total_candidates)
col2.metric("Placement Rate (%)", round(placement_rate,2))
col3.metric("Job Acceptance Rate (%)", round(job_acceptance_rate,2))


col4, col5, col6 = st.columns(3)

col4.metric("Average Interview Score", round(avg_interview_score,2))
col5.metric("Average Skills Match %", round(avg_skills_match,2))
col6.metric("Offer Dropout Rate %", round(offer_dropout_rate,2))


st.metric("High Risk Candidate %", round(high_risk_candidates,2))