
import streamlit as st
import joblib
import numpy as np
from scipy.sparse import hstack, csr_matrix

xgb_model     = joblib.load("model/xgb_model.pkl")
tfidf         = joblib.load("model/tfidf_vectorizer.pkl")
feature_names = joblib.load("model/feature_names.pkl")
explainer     = joblib.load("model/shap_explainer.pkl")

numeric_features = [
    "has_company_logo", "has_questions",
    "text_length", "has_salary", "has_company_profile"
]

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

    sv      = explainer.shap_values(X_new.toarray())[0]
    top_idx = np.argsort(sv)[::-1][:10]

    plain_english = {
        "has_company_logo"   : "No company logo provided",
        "has_questions"      : "No screening questions asked",
        "text_length"        : "Description is unusually short",
        "has_salary"         : "Salary range not specified",
        "has_company_profile": "Missing company profile",
    }

    reasons = []
    for idx in top_idx:
        if sv[idx] > 0:
            fname = feature_names[idx]
            label = plain_english.get(fname, f"Suspicious keyword: {fname}")
            if label not in reasons:
                reasons.append(label)
        if len(reasons) == 3:
            break

    if len(reasons) < 2:
        if has_logo == 0:   reasons.append("No company logo provided")
        if has_salary == 0: reasons.append("Salary range not specified")
        if has_company == 0:reasons.append("Missing company profile")
        if text_length < 300: reasons.append("Description is unusually short")

    return round(float(fraud_prob), 3), risk_label, reasons[:3]


st.set_page_config(page_title="Fake Job Detector", page_icon="🕵️")
st.title("🕵️ Fake Job Listing Detector")
st.markdown("Paste a job listing below to check if it looks suspicious.")

col1, col2 = st.columns(2)
if col1.button("Try a Fake Listing"):
    st.session_state["title"]        = "Work From Home Data Entry - Earn $500/day"
    st.session_state["company"]      = ""
    st.session_state["description"]  = "Easy work from home. No experience needed. Apply now. Immediate joining."
    st.session_state["requirements"] = ""
    st.session_state["salary"]       = ""
    st.session_state["logo"]         = False
    st.session_state["questions"]    = False

if col2.button("Try a Real Listing"):
    st.session_state["title"]        = "Senior Software Engineer - Backend Python"
    st.session_state["company"]      = "Fintech startup based in Bangalore, Series B funded, 200+ employees."
    st.session_state["description"]  = "Looking for backend engineer with 3+ years in Python, FastAPI, PostgreSQL."
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
has_logo     = col3.checkbox("Has Company Logo",       value=st.session_state.get("logo", False))
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
