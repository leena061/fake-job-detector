
import streamlit as st
import joblib
import numpy as np
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

    found_keywords = []
    for kw in FRAUD_KEYWORDS:
        if kw in combined_lower:
            found_keywords.append(f"Suspicious phrase detected: \"{kw}\"")
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

    fraud_prob  = xgb_model.predict_proba(X_new)[0][1]

    if fraud_prob < 0.35:
        risk_label = "Low Risk"
        color      = "#1D9E75"
        icon       = "✅"
    elif fraud_prob < 0.65:
        risk_label = "Medium Risk"
        color      = "#F5A623"
        icon       = "⚠️"
    else:
        risk_label = "High Risk"
        color      = "#E24B4A"
        icon       = "🚨"

    reasons = get_red_flags(
        title, company_profile, description, requirements,
        has_logo, has_questions, salary, text_length
    )

    return round(float(fraud_prob), 3), risk_label, color, icon, reasons


# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Fake Job Detector",
    page_icon="🕵️",
    layout="centered"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    /* Force light background */
    .stApp { background-color: #F8F9FB; }

    /* Header */
    .header-box {
        background: linear-gradient(135deg, #1D3557 0%, #457B9D 100%);
        padding: 2rem 2rem 1.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .header-box h1 {
        color: white !important;
        font-size: 2rem !important;
        margin: 0 !important;
        font-weight: 700 !important;
    }
    .header-box p {
        color: #A8D8EA !important;
        margin: 0.4rem 0 0 0 !important;
        font-size: 1rem !important;
    }

    /* Card */
    .card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        margin-bottom: 1.2rem;
    }

    /* Result box */
    .result-box {
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
        text-align: center;
    }
    .result-score {
        font-size: 3rem;
        font-weight: 800;
        margin: 0;
    }
    .result-label {
        font-size: 1.3rem;
        font-weight: 600;
        margin: 0.2rem 0 0 0;
    }

    /* Red flag item */
    .flag-item {
        background: #FFF3F3;
        border-left: 4px solid #E24B4A;
        border-radius: 0 8px 8px 0;
        padding: 0.6rem 1rem;
        margin-bottom: 0.5rem;
        color: #333;
        font-size: 0.95rem;
    }

    /* Clean flag item */
    .clean-item {
        background: #F0FBF6;
        border-left: 4px solid #1D9E75;
        border-radius: 0 8px 8px 0;
        padding: 0.6rem 1rem;
        color: #333;
        font-size: 0.95rem;
    }

    /* Sample buttons */
    .stButton > button {
        border-radius: 8px !important;
        font-weight: 500 !important;
        border: 1.5px solid #1D3557 !important;
        color: #1D3557 !important;
        background: white !important;
        transition: all 0.2s !important;
    }
    .stButton > button:hover {
        background: #1D3557 !important;
        color: white !important;
    }

    /* Analyse button */
    .stButton > button[kind="primary"] {
        background: #1D3557 !important;
        color: white !important;
        border: none !important;
        padding: 0.6rem 2rem !important;
        font-size: 1rem !important;
    }

    /* Input labels */
    label { color: #1D3557 !important; font-weight: 500 !important; }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────
st.markdown("""
<div class="header-box">
    <h1>🕵️ Fake Job Listing Detector</h1>
    <p>Powered by XGBoost · Trained on 17,880 real-world job postings</p>
</div>
""", unsafe_allow_html=True)

# ── Sample buttons ───────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("**Try a sample listing:**")
col1, col2 = st.columns(2)

if col1.button("🚨 Try a Fake Listing"):
    st.session_state["title"]        = "Work From Home Data Entry - Earn $500/day"
    st.session_state["company"]      = ""
    st.session_state["description"]  = "Easy work from home. No experience needed. Apply now. Immediate joining. Weekly payment via wire transfer. Limited seats available."
    st.session_state["requirements"] = ""
    st.session_state["salary"]       = ""
    st.session_state["logo"]         = False
    st.session_state["questions"]    = False

if col2.button("✅ Try a Real Listing"):
    st.session_state["title"]        = "Senior Software Engineer - Backend Python"
    st.session_state["company"]      = "Fintech startup based in Bangalore, Series B funded, 200+ employees."
    st.session_state["description"]  = "Looking for backend engineer with 3+ years in Python, FastAPI, PostgreSQL. You will design microservices for our payments platform."
    st.session_state["requirements"] = "BSc Computer Science. Strong knowledge of REST APIs, SQL, AWS."
    st.session_state["salary"]       = "18-25 LPA"
    st.session_state["logo"]         = True
    st.session_state["questions"]    = True

st.markdown('</div>', unsafe_allow_html=True)

# ── Input form ───────────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("**📋 Job Listing Details**")

title        = st.text_input("Job Title",       value=st.session_state.get("title", ""), placeholder="e.g. Software Engineer - Backend")
company      = st.text_area("Company Profile",  value=st.session_state.get("company", ""), height=80,  placeholder="Describe the company...")
description  = st.text_area("Job Description",  value=st.session_state.get("description", ""), height=120, placeholder="Paste the job description here...")
requirements = st.text_area("Requirements",     value=st.session_state.get("requirements", ""), height=80, placeholder="Skills and qualifications required...")
salary       = st.text_input("Salary Range",    value=st.session_state.get("salary", ""), placeholder="e.g. 10-15 LPA or $60,000/year")

col3, col4   = st.columns(2)
has_logo     = col3.checkbox("Has Company Logo",        value=st.session_state.get("logo", False))
has_ques     = col4.checkbox("Has Screening Questions", value=st.session_state.get("questions", False))
st.markdown('</div>', unsafe_allow_html=True)

# ── Analyse button ───────────────────────────────────────────
col_btn = st.columns([1, 2, 1])
with col_btn[1]:
    analyse = st.button("🔍 Analyse Listing", type="primary", use_container_width=True)

# ── Results ──────────────────────────────────────────────────
if analyse:
    if not title and not description:
        st.warning("Please enter at least a job title and description.")
    else:
        with st.spinner("Analysing listing..."):
            prob, risk, color, icon, reasons = predict(
                title, company, description, requirements,
                int(has_logo), int(has_ques), salary
            )

        st.markdown(f"""
        <div class="result-box" style="background:{color}18; border: 2px solid {color};">
            <p class="result-score" style="color:{color};">{icon} {prob*100:.1f}%</p>
            <p class="result-label" style="color:{color};">{risk}</p>
            <p style="color:#666; font-size:0.9rem; margin-top:0.5rem;">Fraud Probability Score</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if reasons:
            st.markdown("**⚠️ Red Flags Detected:**")
            for r in reasons:
                st.markdown(f'<div class="flag-item">⚑ {r}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="clean-item">✅ No major red flags detected in this listing</div>', unsafe_allow_html=True)

        # Meter
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Fraud Risk Meter:**")
        st.progress(prob)
