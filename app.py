
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
            found_keywords.append(f'Suspicious phrase detected: \"{kw}\"')
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
        risk_label = "Low Risk"
        color      = "#00C896"
        bg_color   = "rgba(0, 200, 150, 0.1)"
        border     = "rgba(0, 200, 150, 0.4)"
        icon       = "✅"
    elif fraud_prob < 0.65:
        risk_label = "Medium Risk"
        color      = "#FFB347"
        bg_color   = "rgba(255, 179, 71, 0.1)"
        border     = "rgba(255, 179, 71, 0.4)"
        icon       = "⚠️"
    else:
        risk_label = "High Risk"
        color      = "#FF5C5C"
        bg_color   = "rgba(255, 92, 92, 0.1)"
        border     = "rgba(255, 92, 92, 0.4)"
        icon       = "🚨"

    reasons = get_red_flags(
        title, company_profile, description, requirements,
        has_logo, has_questions, salary, text_length
    )

    return round(float(fraud_prob), 3), risk_label, color, bg_color, border, icon, reasons


# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Fake Job Detector",
    page_icon="🕵️",
    layout="centered"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Reset & base ── */
    .stApp { font-family: 'Inter', sans-serif; }

    /* ── Header ── */
    .header-box {
        padding: 2.5rem 2rem 2rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.8rem;
        text-align: center;
        border: 1px solid rgba(99, 102, 241, 0.3);
        background: linear-gradient(135deg, rgba(99,102,241,0.15) 0%, rgba(139,92,246,0.1) 100%);
        position: relative;
        overflow: hidden;
    }
    .header-box::before {
        content: "";
        position: absolute;
        top: -60px; right: -60px;
        width: 180px; height: 180px;
        border-radius: 50%;
        background: rgba(99,102,241,0.08);
    }
    .header-tag {
        display: inline-block;
        background: rgba(99,102,241,0.2);
        color: #A5B4FC;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        border: 1px solid rgba(99,102,241,0.35);
        margin-bottom: 0.8rem;
    }
    .header-box h1 {
        font-size: 2rem !important;
        font-weight: 700 !important;
        margin: 0 0 0.4rem 0 !important;
        letter-spacing: -0.02em;
    }
    .header-box p {
        font-size: 0.9rem !important;
        opacity: 0.6;
        margin: 0 !important;
    }

    /* ── Section label ── */
    .section-label {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        opacity: 0.45;
        margin-bottom: 0.6rem;
    }

    /* ── Sample buttons ── */
    .stButton > button {
        border-radius: 10px !important;
        font-weight: 500 !important;
        font-size: 0.875rem !important;
        transition: all 0.2s !important;
        padding: 0.5rem 1.2rem !important;
    }

    /* ── Analyse button ── */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #6366F1, #8B5CF6) !important;
        color: white !important;
        border: none !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.01em !important;
        padding: 0.65rem 2rem !important;
        box-shadow: 0 4px 15px rgba(99,102,241,0.35) !important;
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 6px 20px rgba(99,102,241,0.5) !important;
        transform: translateY(-1px) !important;
    }

    /* ── Result box ── */
    .result-box {
        border-radius: 16px;
        padding: 2rem;
        margin-top: 1rem;
        text-align: center;
    }
    .result-pct {
        font-size: 3.5rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        line-height: 1;
        margin: 0;
    }
    .result-label {
        font-size: 1.1rem;
        font-weight: 600;
        margin: 0.4rem 0 0 0;
        opacity: 0.85;
    }
    .result-sub {
        font-size: 0.8rem;
        opacity: 0.5;
        margin-top: 0.3rem;
    }

    /* ── Flag items ── */
    .flag-item {
        border-radius: 10px;
        padding: 0.7rem 1rem;
        margin-bottom: 0.5rem;
        font-size: 0.875rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(255, 92, 92, 0.08);
        border: 1px solid rgba(255, 92, 92, 0.2);
    }
    .clean-item {
        border-radius: 10px;
        padding: 0.7rem 1rem;
        font-size: 0.875rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(0, 200, 150, 0.08);
        border: 1px solid rgba(0, 200, 150, 0.2);
    }

    /* ── Progress meter ── */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #6366F1, #8B5CF6, #EC4899) !important;
        border-radius: 999px !important;
    }
    .stProgress > div > div > div {
        border-radius: 999px !important;
        height: 8px !important;
    }

    /* ── Divider ── */
    hr { opacity: 0.15 !important; }

    /* ── Hide Streamlit branding ── */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────
st.markdown("""
<div class="header-box">
    <div class="header-tag">AI-Powered Detection</div>
    <h1>🕵️ Fake Job Listing Detector</h1>
    <p>Powered by XGBoost &nbsp;·&nbsp; Trained on 17,880 real-world job postings</p>
</div>
""", unsafe_allow_html=True)

# ── Sample buttons ───────────────────────────────────────────
st.markdown('<p class="section-label">Try a sample listing</p>', unsafe_allow_html=True)
col1, col2 = st.columns(2)

if col1.button("🚨 Try a Fake Listing", use_container_width=True):
    st.session_state["title"]        = "Work From Home Data Entry - Earn $500/day"
    st.session_state["company"]      = ""
    st.session_state["description"]  = "Easy work from home. No experience needed. Apply now. Immediate joining. Weekly payment via wire transfer. Limited seats available."
    st.session_state["requirements"] = ""
    st.session_state["salary"]       = ""
    st.session_state["logo"]         = False
    st.session_state["questions"]    = False

if col2.button("✅ Try a Real Listing", use_container_width=True):
    st.session_state["title"]        = "Senior Software Engineer - Backend Python"
    st.session_state["company"]      = "Fintech startup based in Bangalore, Series B funded, 200+ employees."
    st.session_state["description"]  = "Looking for backend engineer with 3+ years in Python, FastAPI, PostgreSQL. You will design microservices for our payments platform."
    st.session_state["requirements"] = "BSc Computer Science. Strong knowledge of REST APIs, SQL, AWS."
    st.session_state["salary"]       = "18-25 LPA"
    st.session_state["logo"]         = True
    st.session_state["questions"]    = True

st.markdown("<br>", unsafe_allow_html=True)

# ── Input form ───────────────────────────────────────────────
st.markdown('<p class="section-label">📋 Job Listing Details</p>', unsafe_allow_html=True)

title        = st.text_input("Job Title",        value=st.session_state.get("title", ""),        placeholder="e.g. Software Engineer - Backend")
company      = st.text_area("Company Profile",   value=st.session_state.get("company", ""),      height=90,  placeholder="Describe the company...")
description  = st.text_area("Job Description",   value=st.session_state.get("description", ""),  height=130, placeholder="Paste the job description here...")
requirements = st.text_area("Requirements",      value=st.session_state.get("requirements", ""), height=90,  placeholder="Skills and qualifications required...")
salary       = st.text_input("Salary Range",     value=st.session_state.get("salary", ""),       placeholder="e.g. 10-15 LPA or $60,000/year")

col3, col4 = st.columns(2)
has_logo   = col3.checkbox("Has Company Logo",        value=st.session_state.get("logo", False))
has_ques   = col4.checkbox("Has Screening Questions", value=st.session_state.get("questions", False))

st.markdown("<br>", unsafe_allow_html=True)

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
            prob, risk, color, bg_color, border, icon, reasons = predict(
                title, company, description, requirements,
                int(has_logo), int(has_ques), salary
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="result-box" style="background:{bg_color}; border: 1px solid {border};">
            <p class="result-pct" style="color:{color};">{prob*100:.1f}%</p>
            <p class="result-label" style="color:{color};">{icon} {risk}</p>
            <p class="result-sub">Fraud Probability Score</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Fraud Risk Meter**")
        st.progress(prob)

        st.markdown("<br>", unsafe_allow_html=True)

        if reasons:
            st.markdown('<p class="section-label">Red Flags Detected</p>', unsafe_allow_html=True)
            for r in reasons:
                st.markdown(f'<div class="flag-item">⚑ {r}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="clean-item">✅ No major red flags detected in this listing</div>', unsafe_allow_html=True)
