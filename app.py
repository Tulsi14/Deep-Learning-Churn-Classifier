import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnSense AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0a0f !important;
    color: #e8e6f0 !important;
    font-family: 'Syne', sans-serif;
}

[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse at 20% 10%, #1a0a2e 0%, #0a0a0f 50%, #0d1117 100%) !important;
}

#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

.hero-wrapper {
    text-align: center;
    padding: 3rem 1rem 1.5rem;
}
.hero-badge {
    display: inline-block;
    background: linear-gradient(135deg, #7c3aed22, #06b6d422);
    border: 1px solid #7c3aed55;
    color: #a78bfa;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    padding: 0.35rem 1rem;
    border-radius: 2rem;
    margin-bottom: 1.2rem;
}
.hero-title {
    font-size: clamp(2.8rem, 7vw, 5rem);
    font-weight: 800;
    line-height: 1.05;
    background: linear-gradient(135deg, #ffffff 0%, #a78bfa 45%, #22d3ee 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.8rem;
    letter-spacing: -0.03em;
}
.hero-sub {
    font-size: 1.05rem;
    color: #94a3b8;
    font-weight: 400;
    max-width: 520px;
    margin: 0 auto 2rem;
    line-height: 1.7;
}
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #7c3aed44, #22d3ee44, transparent);
    margin: 0 0 2.5rem;
}
.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #7c3aed;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, #7c3aed33, transparent);
}
.input-section {
    background: linear-gradient(135deg, #13111f 0%, #0f0e1a 100%);
    border: 1px solid #2a2640;
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
    position: relative;
    overflow: hidden;
}
.input-section::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #7c3aed, #22d3ee);
}

label[data-testid="stWidgetLabel"] p {
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    color: #c4b5fd !important;
    letter-spacing: 0.02em !important;
}

div.stButton > button {
    width: 100%;
    padding: 1rem 2rem !important;
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    cursor: pointer !important;
    box-shadow: 0 0 30px #7c3aed44 !important;
    margin-top: 0.5rem;
}
div.stButton > button:hover {
    box-shadow: 0 0 50px #7c3aed88 !important;
    background: linear-gradient(135deg, #8b5cf6, #6366f1) !important;
}

.result-card {
    border-radius: 20px;
    padding: 2.5rem 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    margin-top: 1rem;
}
.result-card.churn {
    background: linear-gradient(135deg, #2d1212 0%, #1a0a0a 100%);
    border: 1px solid #ef444455;
}
.result-card.safe {
    background: linear-gradient(135deg, #0d2312 0%, #0a1a0f 100%);
    border: 1px solid #22c55e55;
}
.result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
}
.result-card.churn::before { background: linear-gradient(90deg, #ef4444, #f97316); }
.result-card.safe::before  { background: linear-gradient(90deg, #22c55e, #06b6d4); }

.result-icon { font-size: 3.5rem; margin-bottom: 0.5rem; }
.result-verdict {
    font-size: 1.6rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin: 0.5rem 0;
}
.result-card.churn .result-verdict { color: #f87171; }
.result-card.safe  .result-verdict { color: #4ade80; }

.result-prob-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #64748b;
    margin: 1.2rem 0 0.3rem;
}
.result-prob-value {
    font-size: 3rem;
    font-weight: 800;
    font-family: 'Space Mono', monospace;
}
.result-card.churn .result-prob-value { color: #ef4444; }
.result-card.safe  .result-prob-value { color: #22c55e; }

.prob-bar-track {
    width: 100%;
    height: 8px;
    background: #1e1b3a;
    border-radius: 99px;
    margin: 0.8rem 0 0.3rem;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 99px;
}
.info-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.8rem;
    margin-top: 1.2rem;
}
.info-tile {
    background: #13111f;
    border: 1px solid #2a2640;
    border-radius: 12px;
    padding: 1rem;
    text-align: left;
}
.info-tile-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 0.3rem;
}
.info-tile-value {
    font-size: 1.1rem;
    font-weight: 700;
    color: #e8e6f0;
}
.footer {
    text-align: center;
    padding: 2rem 0 1rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #374151;
    letter-spacing: 0.08em;
}
</style>
""", unsafe_allow_html=True)

# ─── Load Model & Artefacts ──────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('model.h5')
    with open('label_encoder_gender.pkl', 'rb') as f:
        le_gender = pickle.load(f)
    with open('onehot_encoder_geo.pkl', 'rb') as f:
        ohe_geo = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, le_gender, ohe_geo, scaler

model, label_encoder_gender, onehot_encoder_geo, scaler = load_assets()

# ─── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrapper">
    <div class="hero-badge">🧠 Neural Network · ANN Classification</div>
    <h1 class="hero-title">ChurnSense AI</h1>
    <p class="hero-sub">
        Predict customer churn with a deep learning model trained on 10,000+ banking records.
        Enter customer attributes below to get an instant risk assessment.
    </p>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

# ─── Layout ─────────────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1.15, 1], gap="large")

with left_col:
    st.markdown('<p class="section-label">01 — Demographics</p>', unsafe_allow_html=True)
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        geography = st.selectbox("🌍 Geography", options=onehot_encoder_geo.categories_[0])
    with c2:
        gender = st.selectbox("👤 Gender", options=label_encoder_gender.classes_)
    age = st.slider("🎂 Age", min_value=18, max_value=92, value=38)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<p class="section-label">02 — Account Details</p>', unsafe_allow_html=True)
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        credit_score = st.number_input("💳 Credit Score", min_value=300, max_value=900, value=650, step=1)
    with c4:
        tenure = st.slider("📅 Tenure (Years)", min_value=0, max_value=10, value=5)
    c5, c6 = st.columns(2)
    with c5:
        balance = st.number_input("💰 Account Balance ($)", min_value=0.0, value=75000.0, step=500.0, format="%.2f")
    with c6:
        estimated_salary = st.number_input("💵 Estimated Salary ($)", min_value=0.0, value=60000.0, step=1000.0, format="%.2f")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<p class="section-label">03 — Products & Membership</p>', unsafe_allow_html=True)
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    c7, c8, c9 = st.columns(3)
    with c7:
        num_of_products = st.slider("📦 # Products", min_value=1, max_value=4, value=2)
    with c8:
        has_cr_card = st.selectbox("💳 Has Credit Card", options=[1, 0], format_func=lambda x: "✅ Yes" if x == 1 else "❌ No")
    with c9:
        is_active_member = st.selectbox("⚡ Active Member", options=[1, 0], format_func=lambda x: "✅ Yes" if x == 1 else "❌ No")
    st.markdown('</div>', unsafe_allow_html=True)

    predict_btn = st.button("⚡  Run Churn Prediction", use_container_width=True)

with right_col:
    st.markdown('<p class="section-label">04 — Prediction Output</p>', unsafe_allow_html=True)

    if not predict_btn:
        st.markdown("""
        <div style="background:#13111f;border:1px dashed #2a2640;border-radius:20px;
                    padding:3rem 2rem;text-align:center;color:#4b5563;">
            <div style="font-size:3rem;margin-bottom:1rem;">🔮</div>
            <p style="font-family:'Space Mono',monospace;font-size:0.8rem;letter-spacing:0.1em;text-transform:uppercase;">
                Awaiting Input
            </p>
            <p style="font-size:0.9rem;margin-top:0.5rem;line-height:1.6;">
                Configure customer attributes on the left and hit<br>
                <strong style="color:#a78bfa;">Run Churn Prediction</strong> to see results.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        input_data = pd.DataFrame({
            'CreditScore':     [credit_score],
            'Gender':          [label_encoder_gender.transform([gender])[0]],
            'Age':             [age],
            'Tenure':          [tenure],
            'Balance':         [balance],
            'NumOfProducts':   [num_of_products],
            'HasCrCard':       [has_cr_card],
            'IsActiveMember':  [is_active_member],
            'EstimatedSalary': [estimated_salary],
        })
        geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
        geo_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
        input_data = pd.concat([input_data.reset_index(drop=True), geo_df], axis=1)
        input_scaled = scaler.transform(input_data)

        prob = float(model.predict(input_scaled, verbose=0)[0][0])
        pct  = round(prob * 100, 1)
        churn = prob > 0.5

        if churn:
            card_class = "churn"
            icon       = "🚨"
            verdict    = "High Churn Risk"
            bar_color  = "linear-gradient(90deg,#ef4444,#f97316)"
            risk_label = "HIGH RISK"
            risk_color = "#ef4444"
            action_msg = "Immediate retention action recommended."
        else:
            card_class = "safe"
            icon       = "✅"
            verdict    = "Low Churn Risk"
            bar_color  = "linear-gradient(90deg,#22c55e,#06b6d4)"
            risk_label = "LOW RISK"
            risk_color = "#22c55e"
            action_msg = "Customer appears stable. Monitor quarterly."

        st.markdown(f"""
        <div class="result-card {card_class}">
            <div class="result-icon">{icon}</div>
            <div class="result-verdict">{verdict}</div>
            <div style="color:#64748b;font-size:0.85rem;margin-top:0.3rem;">{action_msg}</div>
            <div class="result-prob-label">Churn Probability</div>
            <div class="result-prob-value">{pct}%</div>
            <div class="prob-bar-track">
                <div class="prob-bar-fill" style="width:{pct}%;background:{bar_color};"></div>
            </div>
            <div style="display:flex;justify-content:space-between;
                        font-family:'Space Mono',monospace;font-size:0.65rem;color:#4b5563;">
                <span>0%</span><span>50% threshold</span><span>100%</span>
            </div>
            <div style="margin-top:1.2rem;display:inline-block;
                        background:{risk_color}22;border:1px solid {risk_color}55;
                        color:{risk_color};font-family:'Space Mono',monospace;
                        font-size:0.7rem;letter-spacing:0.15em;
                        padding:0.3rem 0.9rem;border-radius:99px;">
                {risk_label}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="info-grid">
            <div class="info-tile">
                <div class="info-tile-label">Geography</div>
                <div class="info-tile-value">{geography}</div>
            </div>
            <div class="info-tile">
                <div class="info-tile-label">Age / Gender</div>
                <div class="info-tile-value">{age} · {gender}</div>
            </div>
            <div class="info-tile">
                <div class="info-tile-label">Credit Score</div>
                <div class="info-tile-value">{credit_score}</div>
            </div>
            <div class="info-tile">
                <div class="info-tile-label">Tenure</div>
                <div class="info-tile-value">{tenure} yr{"s" if tenure != 1 else ""}</div>
            </div>
            <div class="info-tile">
                <div class="info-tile-label">Balance</div>
                <div class="info-tile-value">${balance:,.0f}</div>
            </div>
            <div class="info-tile">
                <div class="info-tile-label">Est. Salary</div>
                <div class="info-tile-value">${estimated_salary:,.0f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        confidence = abs(prob - 0.5) / 0.5 * 100
        conf_label = "Very High" if confidence > 75 else "High" if confidence > 50 else "Moderate" if confidence > 25 else "Low"

        st.markdown(f"""
        <div style="margin-top:1rem;background:#13111f;border:1px solid #2a2640;
                    border-radius:12px;padding:1rem 1.2rem;">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <span style="font-family:'Space Mono',monospace;font-size:0.65rem;
                             letter-spacing:0.15em;text-transform:uppercase;color:#64748b;">
                    Model Confidence
                </span>
                <span style="font-family:'Space Mono',monospace;font-size:0.8rem;
                             color:#a78bfa;font-weight:700;">{conf_label} · {confidence:.0f}%</span>
            </div>
            <div class="prob-bar-track" style="margin-top:0.6rem;">
                <div class="prob-bar-fill"
                     style="width:{confidence}%;background:linear-gradient(90deg,#7c3aed,#22d3ee);"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    ChurnSense AI · Powered by TensorFlow ANN · Built with Streamlit
</div>
""", unsafe_allow_html=True)