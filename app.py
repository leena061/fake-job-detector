
import streamlit as st
import joblib
import numpy as np
import re
from scipy.sparse import hstack, csr_matrix

xgb_model     = joblib.load("model/xgb_model.pkl")
tfidf         = joblib.load("model/tfidf_vectorizer.pkl")
feature_names = joblib.load("model/feature_names.pkl")

numeric_features = [
    "has_company_logo", "has_questions",
    "text_length", "has_salary", "has_company_profile"
]

FRAUD_KEYWORDS = [
    "wire transfer", "western union", "money gram", "no experience needed",
    "work from home", "earn per day", "weekly payment", "immediate joining",
    "limited seats", "apply now", "no experience required", "training provided",
    "earn $", "earn usd", "from comfort", "laptop and internet",
    "guaranteed income", "be your own boss", "unlimited earning",
    "part time earn", "data entry", "copy paste", "form filling"
]

def get_red_flags(title, company_profile, description, requirements,
                  has_logo, has_questions, salary, text_length):
    reasons = []
    combined_lower = f"{title} {description} {requirements}".lower()

    # Rule based structural checks
    if has_logo == 0:
        reasons.append("No company logo provided")
    if len(company_profile.strip()) < 10:
        reasons.append("Missing or very short company profile")
    if salary.strip() == "":
        reasons.append("Salary range not mentioned")
    if has_questions == 0:
        reasons.append("No screening questions asked")
    if text_length < 200:
        reasons.append("Job description is unusually short")

    # Keyword checks
    found_keywords = []
    for kw in FRAUD_KEYWORDS:
        if kw in combined_lower:
            found_keywords.append(f'Suspicious phrase detected: "{kw}"'  )
    reasons.extend(found_keywords[:2])

    return reasons[:3]

def predict(title, company_profile, description, requirements,
            has_logo, has_questions, salary):

    combined    = f"{title} {company_profile} {description} {requirements}"
    text_length = len(combined)
    has_salary  = 1 if salary.strip() != "" else 0
    has_company = 1 if len(company_profile.strip()) > 10 else 0

    X_text  = tfidf.transform([combined])
    X_num   = csr_matrix([[has_logo, has_questions, text_length, has_salary, has_company]])
    X_new   = hstack([X_text, X_num])

    fraud_prob = xgb_model.predict_proba(X_new)[0][1]

    if fraud_prob < 0.35:
        risk_label = "🟢 Low Risk"
    elif fraud_prob < 0.65:
        risk_label = "🟡 Medium Risk"
    else:
        risk_label = "🔴 High Risk"

    reasons = get_red_flags(
        title, company_profile, description, requirements,
        has_logo, has_questions, salary, text_length
    )

    return round(float(fraud_prob), 3), risk_label, reasons


st.set_page_config(page_title="Fake Job Detector", page_icon="🕵️")
st.title("🕵️ Fake Job Listing Detector")
st.markdown("Paste a job listing below to check if it looks suspicious.")

col1, col2 = st.columns(2)
if col1.button("Try a Fake Listing"):
    st.session_state["title"]        = "Work From Home Data Entry - Earn $500/day"
    st.session_state["company"]      = ""
    st.session_state["description"]  = "Easy work from home. No experience needed. Apply now. Immediate joining. Weekly payment via wire transfer."
    st.session_state["requirements"] = ""
    st.session_state["salary"]       = ""
    st.session_state["logo"]         = False
    st.session_state["questions"]    = False

if col2.button("Try a Real Listing"):
    st.session_state["title"]        = "Senior Software Engineer - Backend Python"
    st.session_state["company"]      = "Fintech startup based in Bangalore, Series B funded, 200+ employees."
    st.session_state["description"]  = "Looking for backend engineer with 3+ years in Python, FastAPI, PostgreSQL. You will design microservices for our payments platform."
    st.session_state["requirements"] = "BSc Computer Science. Strong knowledge of REST APIs, SQL, AWS."
    st.session_state["salary"]       = "18-25 LPA"
    st.session_state["logo"]         = True
    st.session_state["questions"]    = True

st.divider()

title        = st.text_input("Job Title",        value=st.session_state.get("title", ""))
company      = st.text_area("Company Profile",   value=st.session_state.get("company", ""), height=80)
description  = st.text_area("Job Description",   value=st.session_state.get("description", ""), height=120)
requirements = st.text_area("Requirements",      value=st.session_state.get("requirements", ""), height=80)
salary       = st.text_input("Salary Range",     value=st.session_state.get("salary", ""))

col3, col4   = st.columns(2)
has_logo     = col3.checkbox("Has Company Logo",        value=st.session_state.get("logo", False))
has_ques     = col4.checkbox("Has Screening Questions", value=st.session_state.get("questions", False))

st.divider()

if st.button("🔍 Analyse Listing", type="primary"):
    if not title and not description:
        st.warning("Please enter at least a job title and description.")
    else:
        with st.spinner("Analysing..."):
            prob, risk, reasons = predict(
                title, company, description, requirements,
                int(has_logo), int(has_ques), salary
            )

        st.markdown(f"### Result: {risk}")
        st.progress(prob)
        st.markdown(f"**Fraud Probability: {prob*100:.1f}%**")

        if reasons:
            st.markdown("### ⚠️ Red Flags Detected:")
            for r in reasons:
                st.markdown(f"- {r}")
        else:
            st.markdown("### ✅ No major red flags detected")
