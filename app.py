import os
import json
import sqlite3
import base64
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib
from PIL import Image

from pipeline import (
    load_pipeline,
    engineer_features,
    preprocess_for_inference,
    compute_unified_risk_score,
    explain_transaction_shap,
    auto_detect_and_map_columns,
    train_fresh_model_on_any_dataset,
    CITY_COORDINATES,
    ARTIFACTS_DIR,
    DATA_PATH,
    train_and_save_pipeline
)
from db import (
    init_db,
    add_case,
    get_all_cases,
    update_case,
    delete_case,
    create_user,
    verify_user
)

LOGO_PATH = os.path.join(os.path.dirname(__file__), "assets", "logo.jpg")

def get_logo_base64():
    if os.path.exists(LOGO_PATH):
        try:
            with open(LOGO_PATH, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            return ""
    return ""

LOGO_B64 = get_logo_base64()

st.set_page_config(
    page_title="FinShield | Autonomous Anomaly Surveillance",
    page_icon=Image.open(LOGO_PATH) if os.path.exists(LOGO_PATH) else "🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Standard Enterprise Light Neutral Gray Design System with Auth Animations
st.markdown("""
<style>
    /* Base Typography & Canvas */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    
    code, pre, .mono {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
    }

    /* Soft Neutral Gray Canvas */
    .stApp {
        background-color: #F1F4F9 !important;
        color: #1E293B !important;
    }

    /* ----------------------------------------------------------------- */
    /* AUTHENTICATION PORTAL & ANIMATIONS                                */
    /* ----------------------------------------------------------------- */
    @keyframes authCardEntrance {
        0% {
            opacity: 0;
            transform: translateY(22px) scale(0.985);
        }
        100% {
            opacity: 1;
            transform: translateY(0) scale(1);
        }
    }

    @keyframes shieldBob {
        0%, 100% {
            transform: translateY(0);
        }
        50% {
            transform: translateY(-4px);
        }
    }

    @keyframes scanLaser {
        0% {
            left: -120%;
        }
        50% {
            left: 120%;
        }
        100% {
            left: -120%;
        }
    }

    @keyframes pulseRingSoft {
        0% {
            box-shadow: 0 0 0 0 rgba(37, 99, 235, 0.25);
        }
        70% {
            box-shadow: 0 0 0 10px rgba(37, 99, 235, 0);
        }
        100% {
            box-shadow: 0 0 0 0 rgba(37, 99, 235, 0);
        }
    }

    .auth-card-wrapper {
        animation: authCardEntrance 0.45s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        background: #FFFFFF;
        border: 1px solid #D8E1EC;
        border-radius: 12px;
        padding: 30px 34px;
        box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.07);
        max-width: 480px;
        margin: 20px auto 30px auto;
    }

    .auth-logo-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 76px;
        height: 76px;
        border-radius: 18px;
        background: #FFFFFF;
        border: 1px solid #BFDBFE;
        margin-bottom: 12px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 4px 16px rgba(37, 99, 235, 0.12);
        animation: shieldBob 3.6s ease-in-out infinite, pulseRingSoft 3s infinite;
    }

    .auth-logo-badge img {
        width: 58px;
        height: 58px;
        object-fit: contain;
        display: block;
    }

    .auth-brand-name {
        font-size: 1.65rem;
        font-weight: 800;
        color: #0F172A;
        margin: 0;
        letter-spacing: -0.02em;
        line-height: 1.15;
    }

    .auth-tagline {
        font-size: 0.82rem;
        font-weight: 700;
        color: #2563EB;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-top: 4px;
        margin-bottom: 4px;
    }

    .auth-subtitle {
        color: #64748B;
        font-size: 0.84rem;
        margin-top: 2px;
        margin-bottom: 16px;
    }

    .auth-user-badge {
        background: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 12px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }

    .sidebar-brand-card {
        background: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-radius: 8px;
        padding: 12px 14px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }

    .sidebar-brand-header {
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .sidebar-brand-img {
        width: 38px;
        height: 38px;
        object-fit: contain;
        border-radius: 8px;
        background: #FFFFFF;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
        border: 1px solid #E2E8F0;
        padding: 2px;
        flex-shrink: 0;
    }

    .sidebar-brand-title {
        font-size: 1.15rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.02em;
        line-height: 1.1;
    }

    .sidebar-brand-badge {
        font-size: 0.64rem;
        font-weight: 700;
        color: #2563EB;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }

    .sidebar-brand-tagline {
        font-size: 0.72rem;
        font-weight: 500;
        color: #64748B;
        margin-top: 8px;
        line-height: 1.35;
        border-top: 1px solid #F1F5F9;
        padding-top: 6px;
    }

    /* ----------------------------------------------------------------- */
    /* DASHBOARD ELEMENTS                                                */
    /* ----------------------------------------------------------------- */
    .op-header {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 18px 22px;
        margin-bottom: 18px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 12px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
    }
    
    .op-breadcrumb {
        font-size: 0.75rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 3px;
    }

    .op-title {
        font-size: 1.55rem;
        font-weight: 700;
        color: #0F172A;
        margin: 0;
        letter-spacing: -0.01em;
        line-height: 1.2;
    }

    .op-subtitle {
        color: #64748B;
        font-size: 0.88rem;
        margin-top: 3px;
        margin-bottom: 0;
    }

    .op-status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 6px 12px;
        font-size: 0.8rem;
        font-weight: 500;
        color: #334155;
    }

    .status-dot {
        width: 7px;
        height: 7px;
        background: #059669;
        border-radius: 50%;
        display: inline-block;
    }

    /* Standard 4-Metric Grid */
    .metric-grid-4 {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-bottom: 18px;
    }

    @media (max-width: 1100px) {
        .metric-grid-4 {
            grid-template-columns: repeat(2, 1fr);
        }
    }

    @media (max-width: 640px) {
        .metric-grid-4 {
            grid-template-columns: 1fr;
        }
    }

    .op-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px 20px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }

    .op-card:hover {
        border-color: #CBD5E1;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
    }

    .op-card-label {
        font-size: 0.74rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 4px;
    }

    .op-card-val {
        font-size: 1.75rem;
        font-weight: 700;
        color: #0F172A;
        line-height: 1.2;
        font-feature-settings: "tnum";
        font-variant-numeric: tabular-nums;
    }

    .op-card-context {
        font-size: 0.8rem;
        color: #64748B;
        margin-top: 5px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    /* Clean Card Container for Plots */
    .panel-box {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px 18px;
        margin-bottom: 16px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
    }

    .panel-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #0F172A;
        margin-bottom: 2px;
    }

    .panel-desc {
        font-size: 0.8rem;
        color: #64748B;
        margin-bottom: 12px;
    }

    /* Standard Sidebar */
    [data-testid="stSidebar"] {
        background-color: #E2E8F0 !important;
        border-right: 1px solid #CBD5E1 !important;
    }

    /* Inputs on Neutral Canvas */
    .stTextInput input, .stNumberInput input, .stSelectbox [data-baseweb="select"] {
        background-color: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 6px !important;
        color: #0F172A !important;
        font-size: 0.88rem !important;
        transition: border-color 0.15s ease, box-shadow 0.15s ease;
    }
    
    .stTextInput input:focus, .stNumberInput input:focus {
        border-color: #2563EB !important;
        box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.12) !important;
    }

    /* Standard Buttons */
    .stButton > button {
        background: #FFFFFF !important;
        color: #0F172A !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        border: 1px solid #CBD5E1 !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04) !important;
        transition: all 0.15s ease !important;
    }

    .stButton > button:hover {
        background: #F8FAFC !important;
        border-color: #94A3B8 !important;
        color: #0F172A !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize database
init_db()

# Session State for Authentication
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user" not in st.session_state:
    st.session_state.user = None

@st.cache_resource
def get_model_pipeline():
    if not os.path.exists(os.path.join(ARTIFACTS_DIR, "isolation_forest.joblib")):
        train_and_save_pipeline()
    return load_pipeline()

@st.cache_data
def get_baseline_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None

try:
    iso_model, lof_model, scaler, metadata, explainer = get_model_pipeline()
    baseline_df = get_baseline_data()
except Exception as e:
    st.error(f"Error loading model pipeline: {e}")
    st.stop()

# Plotly theme helper
def apply_clean_theme(fig, title=None):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(family="-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif", color="#475569", size=11),
        title=dict(
            text=title or fig.layout.title.text or "",
            font=dict(family="-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif", size=13, color="#0F172A")
        ),
        margin=dict(l=15, r=15, t=35, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color="#64748B")
        )
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#F1F5F9", tickfont=dict(size=11, color="#64748B"))
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#F1F5F9", tickfont=dict(size=11, color="#64748B"))
    return fig

# Standard Neutral Risk Gauge
def create_sleek_risk_gauge(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Composite Risk Index", 'font': {'size': 13, 'color': '#475569'}},
        number={'suffix': "%", 'font': {'size': 34, 'color': '#0F172A', 'weight': 'bold'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#CBD5E1", 'tickfont': {'color': '#64748B'}},
            'bar': {'color': "#2563EB", 'thickness': 0.28},
            'bgcolor': "#FFFFFF",
            'borderwidth': 1,
            'bordercolor': "#CBD5E1",
            'steps': [
                {'range': [0, 30], 'color': "rgba(16, 185, 129, 0.2)"},
                {'range': [30, 60], 'color': "rgba(245, 158, 11, 0.2)"},
                {'range': [60, 80], 'color': "rgba(249, 115, 22, 0.25)"},
                {'range': [80, 100], 'color': "rgba(239, 68, 68, 0.3)"}
            ],
            'threshold': {
                'line': {'color': "#DC2626", 'width': 3},
                'thickness': 0.85,
                'value': 75
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        height=240,
        margin=dict(l=20, r=20, t=35, b=15)
    )
    return fig

# =============================================================================
# AUTHENTICATION PORTAL (UNAUTHENTICATED STATE)
# =============================================================================
if not st.session_state.authenticated:
    # Hide sidebar while unauthenticated for a clean gateway experience
    st.markdown("""
    <style>
        [data-testid="stSidebar"] { display: none !important; }
    </style>
    """, unsafe_allow_html=True)

    col_l, col_m, col_r = st.columns([1, 1.35, 1])
    
    with col_m:
        st.markdown(f"""
        <div class="auth-card-wrapper">
            <div style="text-align: center;">
                <div class="auth-logo-badge">
                    <img src="data:image/jpeg;base64,{LOGO_B64}" alt="FinShield Logo" />
                </div>
                <h1 class="auth-brand-name">FinShield</h1>
                <div class="auth-tagline">Autonomous Financial Anomaly Surveillance</div>
                <p class="auth-subtitle">Verify investigator credentials to access banking anomaly intelligence</p>
            </div>
        """, unsafe_allow_html=True)

        auth_tab1, auth_tab2 = st.tabs(["Sign In to Console", "Create Investigator Account"])

        with auth_tab1:
            st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)
            with st.form("login_form"):
                login_user = st.text_input("Username or Work Email", placeholder="e.g. analyst or sarah@bankops.internal")
                login_pass = st.text_input("Password", type="password", placeholder="••••••••")
                submit_login = st.form_submit_button("Authenticate & Enter", use_container_width=True)

                if submit_login:
                    if not login_user or not login_pass:
                        st.warning("Please enter your username and password.")
                    else:
                        user_record, msg = verify_user(login_user, login_pass)
                        if user_record:
                            st.session_state.authenticated = True
                            st.session_state.user = user_record
                            st.success(f"Verified: Welcome back, {user_record['full_name']}!")
                            st.rerun()
                        else:
                            st.error(msg)

            st.markdown("<div style='margin-top: 14px; border-top: 1px solid #E2E8F0; padding-top: 12px;'></div>", unsafe_allow_html=True)
            st.markdown("<div style='font-size: 0.74rem; font-weight: 600; color: #64748B; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 8px;'>Quick Demo Access (1-Click)</div>", unsafe_allow_html=True)
            demo_c1, demo_c2 = st.columns(2)
            
            with demo_c1:
                if st.button("🔑 Sarah Jenkins (Analyst)", use_container_width=True):
                    user_record, _ = verify_user("analyst", "investigator123")
                    if user_record:
                        st.session_state.authenticated = True
                        st.session_state.user = user_record
                        st.rerun()
                        
            with demo_c2:
                if st.button("⚡ Alex Vance (Lead)", use_container_width=True):
                    user_record, _ = verify_user("admin", "admin123")
                    if user_record:
                        st.session_state.authenticated = True
                        st.session_state.user = user_record
                        st.rerun()

        with auth_tab2:
            st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)
            with st.form("signup_form"):
                su_fullname = st.text_input("Full Name", placeholder="e.g. Marcus Reid")
                su_email = st.text_input("Corporate Email", placeholder="e.g. marcus.reid@bankops.internal")
                su_username = st.text_input("Username", placeholder="e.g. mreid")
                su_role = st.selectbox(
                    "Operational Role",
                    [
                        "Fraud Operations Analyst",
                        "Senior Risk Investigator",
                        "Compliance & Audit Officer",
                        "Lead Risk Architect"
                    ]
                )
                su_pass = st.text_input("Password", type="password", placeholder="Minimum 6 characters")
                su_pass_confirm = st.text_input("Confirm Password", type="password", placeholder="Repeat password")
                submit_signup = st.form_submit_button("Register Account", use_container_width=True)

                if submit_signup:
                    if not su_fullname or not su_email or not su_username or not su_pass:
                        st.warning("Please fill in all required registration fields.")
                    elif len(su_pass) < 6:
                        st.warning("Password must be at least 6 characters.")
                    elif su_pass != su_pass_confirm:
                        st.error("Passwords do not match.")
                    else:
                        success, message = create_user(
                            username=su_username,
                            email=su_email,
                            password=su_pass,
                            full_name=su_fullname,
                            role=su_role
                        )
                        if success:
                            st.success(f"{message} Signing you in...")
                            user_rec, _ = verify_user(su_username, su_pass)
                            st.session_state.authenticated = True
                            st.session_state.user = user_rec
                            st.rerun()
                        else:
                            st.error(message)

        st.markdown("</div>", unsafe_allow_html=True)

# =============================================================================
# AUTHENTICATED DASHBOARD
# =============================================================================
else:
    user = st.session_state.user or {
        'full_name': 'Sarah Jenkins',
        'role': 'Fraud Operations Analyst',
        'username': 'analyst'
    }

    # Sidebar: FinShield Brand Header
    st.sidebar.markdown(f"""
    <div class="sidebar-brand-card">
        <div class="sidebar-brand-header">
            <img src="data:image/jpeg;base64,{LOGO_B64}" class="sidebar-brand-img" alt="FinShield" />
            <div>
                <div class="sidebar-brand-title">FinShield</div>
                <div class="sidebar-brand-badge">SURVEILLANCE AI</div>
            </div>
        </div>
        <div class="sidebar-brand-tagline">Autonomous Financial Anomaly Surveillance</div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar: User Profile & Controls
    st.sidebar.markdown(f"""
    <div class="auth-user-badge">
        <div style="font-size: 0.7rem; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.05em;">ACTIVE OPERATOR</div>
        <div style="font-size: 0.95rem; font-weight: 700; color: #0F172A; margin-top: 2px;">{user['full_name']}</div>
        <div style="font-size: 0.76rem; color: #2563EB; font-weight: 600;">{user['role']}</div>
        <div style="display: flex; align-items: center; gap: 5px; margin-top: 6px; font-size: 0.72rem; color: #059669;">
            <span class="status-dot"></span> Session Verified
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.sidebar.button("🚪 Sign Out", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.user = None
        st.rerun()

    st.sidebar.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)

    nav = st.sidebar.radio(
        "NAVIGATION",
        [
            "Surveillance & Anomaly Overview",
            "Transaction Risk Evaluator",
            "Universal Dataset Auditor",
            "Fraud Case Management Queue",
            "Model Governance & API"
        ]
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("<div style='font-size: 0.72rem; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px;'>MODEL CONTROLS</div>", unsafe_allow_html=True)
    sensitivity = st.sidebar.slider(
        "Contamination Threshold (% Sensitivity)",
        min_value=1.0,
        max_value=15.0,
        value=5.0,
        step=0.5,
        help="Determines the percentage threshold for classifying statistical outliers."
    )

    st.sidebar.markdown(f"""
    <div style='background: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 6px; padding: 10px 12px; margin-top: 12px; font-size: 0.78rem;'>
        <div style='font-weight: 600; color: #334155; margin-bottom: 4px;'>Pipeline Configuration</div>
        <div style='color: #64748B;'>• Isolation Forest (100 Trees)</div>
        <div style='color: #64748B;'>• Local Outlier Factor (k=20)</div>
        <div style='color: #059669; font-weight: 600; margin-top: 4px;'>● Active Ingestion Stream</div>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------
    # PAGE 1: SURVEILLANCE & ANOMALY OVERVIEW
    # -------------------------------------------------------------
    if nav == "Surveillance & Anomaly Overview":
        # Standard Professional Header with FinShield Branding
        st.markdown(f"""
        <div class="op-header">
            <div style="display: flex; align-items: center; gap: 14px;">
                <img src="data:image/jpeg;base64,{LOGO_B64}" style="width: 46px; height: 46px; object-fit: contain; border-radius: 8px; border: 1px solid #E2E8F0; background: #FFFFFF; padding: 2px;" alt="FinShield" />
                <div>
                    <div class="op-breadcrumb">FinShield / Anomaly Surveillance / Command Overview</div>
                    <h1 class="op-title">Surveillance Command &amp; Heat Matrix</h1>
                    <p class="op-subtitle">Autonomous Financial Anomaly Surveillance &amp; Dual-Model Behavioral Fraud Intelligence.</p>
                </div>
            </div>
            <div style="display: flex; gap: 8px; align-items: center;">
                <div class="op-status-badge">
                    <span class="status-dot"></span> Surveillance Active
                </div>
                <div class="op-status-badge">
                    Operator: {user['full_name']}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if baseline_df is not None:
            with st.spinner("Evaluating transaction dataset..."):
                enriched_df = engineer_features(baseline_df, upper_limit=metadata['upper_limit'])
                X_eval = preprocess_for_inference(enriched_df, scaler, metadata)
                
                iso_scores = iso_model.decision_function(X_eval)
                lof_scores = lof_model.score_samples(X_eval)
                
                threshold_cutoff = np.percentile(iso_scores, sensitivity)
                iso_preds = np.where(iso_scores < threshold_cutoff, -1, 1)
                lof_preds = np.where(lof_scores < np.percentile(lof_scores, sensitivity), -1, 1)
                
                risk_pcts = [compute_unified_risk_score(s, l) for s, l in zip(iso_scores, lof_scores)]
                enriched_df['Risk_Score_Pct'] = risk_pcts
                enriched_df['iso_anomaly'] = np.where(iso_preds == -1, 'Potential Fraud', 'Normal')
                enriched_df['lof_anomaly'] = np.where(lof_preds == -1, 'Potential Fraud', 'Normal')
                
                enriched_df['risk_category'] = 'Normal'
                enriched_df.loc[
                    (enriched_df['iso_anomaly'] == 'Potential Fraud') | (enriched_df['lof_anomaly'] == 'Potential Fraud'),
                    'risk_category'
                ] = 'Single Model Flag'
                enriched_df.loc[
                    (enriched_df['iso_anomaly'] == 'Potential Fraud') & (enriched_df['lof_anomaly'] == 'Potential Fraud'),
                    'risk_category'
                ] = 'Critical Consensus'

            # Interactive Filter Toolbar
            st.markdown("<div style='font-size: 0.74rem; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px;'>SURVEILLANCE FILTERS</div>", unsafe_allow_html=True)
            filt_col1, filt_col2, filt_col3, filt_col4 = st.columns([1.2, 1.2, 1.4, 2])
            
            with filt_col1:
                sel_channel = st.selectbox("Payment Channel", ["All Channels"] + metadata.get("unique_channels", ["Online", "ATM", "Branch"]), label_visibility="collapsed")
            with filt_col2:
                sel_type = st.selectbox("Transaction Type", ["All Types"] + metadata.get("unique_types", ["Debit", "Credit"]), label_visibility="collapsed")
            with filt_col3:
                sel_risk = st.selectbox("Risk Classification", ["All Records", "Critical Consensus (Both)", "Single Engine Flag", "Normal Only"], label_visibility="collapsed")
            with filt_col4:
                search_query = st.text_input("Search Account / Transaction ID", placeholder="Filter by ID (e.g. AC00128, TX000001)...", label_visibility="collapsed")

            # Apply Filters
            filtered_df = enriched_df.copy()
            if sel_channel != "All Channels":
                filtered_df = filtered_df[filtered_df['channel'] == sel_channel]
            if sel_type != "All Types":
                filtered_df = filtered_df[filtered_df['transaction_type'] == sel_type]
            if sel_risk == "Critical Consensus (Both)":
                filtered_df = filtered_df[filtered_df['risk_category'] == 'Critical Consensus']
            elif sel_risk == "Single Engine Flag":
                filtered_df = filtered_df[filtered_df['risk_category'] == 'Single Model Flag']
            elif sel_risk == "Normal Only":
                filtered_df = filtered_df[filtered_df['risk_category'] == 'Normal']
            if search_query:
                sq = search_query.strip().lower()
                filtered_df = filtered_df[
                    filtered_df['account_id'].str.lower().str.contains(sq, na=False) |
                    filtered_df['transaction_id'].str.lower().str.contains(sq, na=False)
                ]

            # Standard 4-Metric Grid (Words Fit Cleanly and Naturally)
            total_tx = len(filtered_df)
            total_accounts = filtered_df['account_id'].nunique()
            both_count = (filtered_df['risk_category'] == 'Critical Consensus').sum()
            flagged_count = (filtered_df['risk_category'] != 'Normal').sum()
            flagged_rate = (flagged_count / total_tx * 100) if total_tx > 0 else 0
            high_risk_volume = filtered_df[filtered_df['Risk_Score_Pct'] >= 70]['transaction_amount'].sum()

            st.markdown(f"""
            <div class="metric-grid-4">
                <div class="op-card">
                    <div class="op-card-label">TOTAL TRANSACTIONS</div>
                    <div class="op-card-val">{total_tx:,}</div>
                    <div class="op-card-context">{total_accounts:,} accounts evaluated</div>
                </div>
                <div class="op-card">
                    <div class="op-card-label">ANOMALIES FLAGGED</div>
                    <div class="op-card-val" style="color: #D97706;">{flagged_count:,}</div>
                    <div class="op-card-context">{flagged_rate:.1f}% portfolio review rate</div>
                </div>
                <div class="op-card">
                    <div class="op-card-label">DUAL-MODEL CONSENSUS</div>
                    <div class="op-card-val" style="color: #DC2626;">{both_count:,}</div>
                    <div class="op-card-context">Both IsoForest & LOF agreed</div>
                </div>
                <div class="op-card">
                    <div class="op-card-label">HIGH-RISK EXPOSURE</div>
                    <div class="op-card-val">${high_risk_volume:,.0f}</div>
                    <div class="op-card-context">Cumulative flagged volume</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # SECTION 1: STATISTICAL DISTRIBUTION & CONSENSUS MATRIX
            sec1_c1, sec1_c2 = st.columns([1.35, 1])

            with sec1_c1:
                st.markdown("""
                <div class="panel-box">
                    <div class="panel-title">Transaction Amount Distribution & Statistical Boundaries</div>
                    <div class="panel-desc">Visualizes transaction volume across amount bands, overlaid with empirical IQR threshold.</div>
                """, unsafe_allow_html=True)
                
                fig_amt = px.histogram(
                    filtered_df,
                    x='transaction_amount',
                    color='risk_category',
                    nbins=45,
                    color_discrete_map={
                        'Normal': '#059669',
                        'Single Model Flag': '#D97706',
                        'Critical Consensus': '#DC2626'
                    },
                    barmode='overlay',
                    labels={'transaction_amount': 'Transaction Amount ($)', 'risk_category': 'Classification', 'count': 'Transactions'}
                )
                fig_amt.add_vline(
                    x=metadata['upper_limit'],
                    line_dash="dash",
                    line_color="#DC2626",
                    annotation_text=f"IQR Threshold (${metadata['upper_limit']:.0f})",
                    annotation_position="top right"
                )
                apply_clean_theme(fig_amt)
                fig_amt.update_layout(height=320)
                st.plotly_chart(fig_amt, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with sec1_c2:
                st.markdown("""
                <div class="panel-box">
                    <div class="panel-title">Detection Consensus Matrix (Agreement Rate)</div>
                    <div class="panel-desc">Compares Isolation Forest decisions against Local Outlier Factor density classifications.</div>
                """, unsafe_allow_html=True)

                ct = pd.crosstab(filtered_df['iso_anomaly'], filtered_df['lof_anomaly'])
                total_cases = len(filtered_df)
                agreement_cases = (
                    (filtered_df['iso_anomaly'] == filtered_df['lof_anomaly']).sum()
                )
                agreement_pct = (agreement_cases / total_cases * 100) if total_cases > 0 else 100.0

                fig_ct = px.imshow(
                    ct,
                    text_auto=True,
                    color_continuous_scale=[[0, '#F8FAFC'], [0.5, '#BFDBFE'], [1, '#2563EB']],
                    labels=dict(x="Local Outlier Factor (LOF)", y="Isolation Forest", color="Count")
                )
                apply_clean_theme(fig_ct)
                fig_ct.update_layout(height=265, coloraxis_showscale=False)
                st.plotly_chart(fig_ct, use_container_width=True)
                
                st.markdown(f"<div style='font-size: 0.82rem; color: #475569; text-align: center;'><b>Model Consensus Rate:</b> {agreement_pct:.1f}% ({agreement_cases:,} / {total_cases:,} transactions)</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            # SECTION 2: CHANNEL & DEMOGRAPHIC RISK CONCENTRATION
            sec2_c1, sec2_c2 = st.columns(2)

            with sec2_c1:
                st.markdown("""
                <div class="panel-box">
                    <div class="panel-title">Risk Distribution by Payment Channel</div>
                    <div class="panel-desc">Comparison of normal transaction volume versus flagged anomalies across ATM, Online, and Branch rails.</div>
                """, unsafe_allow_html=True)

                ch_summary = filtered_df.groupby(['channel', 'risk_category']).size().reset_index(name='count')
                fig_ch = px.bar(
                    ch_summary,
                    x='channel',
                    y='count',
                    color='risk_category',
                    barmode='group',
                    color_discrete_map={
                        'Normal': '#059669',
                        'Single Model Flag': '#D97706',
                        'Critical Consensus': '#DC2626'
                    },
                    labels={'channel': 'Payment Channel', 'count': 'Transactions', 'risk_category': 'Status'}
                )
                apply_clean_theme(fig_ch)
                fig_ch.update_layout(height=290)
                st.plotly_chart(fig_ch, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with sec2_c2:
                st.markdown("""
                <div class="panel-box">
                    <div class="panel-title">Suspicious Volume by Customer Occupation</div>
                    <div class="panel-desc">Identifies income and occupation demographic groups exhibiting anomalous transaction velocity or amounts.</div>
                """, unsafe_allow_html=True)

                flagged_only = filtered_df[filtered_df['risk_category'] != 'Normal']
                if not flagged_only.empty:
                    occ_summary = flagged_only.groupby('customer_occupation').size().reset_index(name='flagged_count').sort_values('flagged_count', ascending=True)
                    fig_occ = px.bar(
                        occ_summary,
                        x='flagged_count',
                        y='customer_occupation',
                        orientation='h',
                        color_discrete_sequence=['#2563EB'],
                        labels={'customer_occupation': 'Occupation', 'flagged_count': 'Flagged Transactions'}
                    )
                    apply_clean_theme(fig_occ)
                    fig_occ.update_layout(height=290)
                    st.plotly_chart(fig_occ, use_container_width=True)
                else:
                    st.info("No flagged transactions match current filter selection.")
                st.markdown("</div>", unsafe_allow_html=True)

            # SECTION 3: GEOGRAPHIC RISK DISTRIBUTION
            st.markdown("""
            <div class="panel-box">
                <div class="panel-title">Geographic Anomaly Distribution (Metropolitan Areas)</div>
                <div class="panel-desc">Visualizes anomalous banking movements across U.S. cities. Sized by transaction amount and colored by risk classification.</div>
            """, unsafe_allow_html=True)

            map_data = filtered_df.dropna(subset=['latitude', 'longitude']).copy()
            if not map_data.empty:
                fig_map = px.scatter_geo(
                    map_data,
                    lat='latitude',
                    lon='longitude',
                    color='risk_category',
                    size='transaction_amount',
                    hover_name='location',
                    hover_data={
                        'transaction_id': True,
                        'account_id': True,
                        'transaction_amount': ':.2f',
                        'Risk_Score_Pct': ':.1f',
                        'latitude': False,
                        'longitude': False
                    },
                    color_discrete_map={
                        'Normal': '#059669',
                        'Single Model Flag': '#D97706',
                        'Critical Consensus': '#DC2626'
                    },
                    scope='usa'
                )
                fig_map.update_geos(
                    bgcolor="#FFFFFF",
                    lakecolor="#F1F4F9",
                    landcolor="#F8FAFC",
                    subunitcolor="#E2E8F0",
                    countrycolor="#CBD5E1",
                    showsubunits=True,
                    showlakes=True
                )
                apply_clean_theme(fig_map)
                fig_map.update_layout(height=430, margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig_map, use_container_width=True)
            else:
                st.info("No geographic coordinates found for the current filtered transactions.")
            st.markdown("</div>", unsafe_allow_html=True)

            # SECTION 4: PRIORITY FLAGGED TRANSACTIONS QUEUE
            st.markdown("""
            <div class="panel-box">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <div>
                        <div class="panel-title">Priority Flagged Transactions Ledger</div>
                        <div class="panel-desc" style="margin-bottom: 0;">Operational queue of high-risk transactions requiring fraud analyst investigation.</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            ledger_display = filtered_df[filtered_df['risk_category'] != 'Normal'].copy()
            if not ledger_display.empty:
                ledger_cols = [
                    'transaction_id', 'account_id', 'transaction_amount', 'account_balance',
                    'channel', 'customer_occupation', 'location', 'Risk_Score_Pct', 'risk_category'
                ]
                ledger_cols_present = [c for c in ledger_cols if c in ledger_display.columns]
                
                st.dataframe(
                    ledger_display[ledger_cols_present].sort_values('Risk_Score_Pct', ascending=False),
                    use_container_width=True,
                    hide_index=True
                )
                
                csv_export = ledger_display[ledger_cols_present].to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Export Filtered Ledger (CSV)",
                    data=csv_export,
                    file_name="flagged_surveillance_ledger.csv",
                    mime="text/csv"
                )
            else:
                st.info("No anomalies detected in the current filter selection.")
                
            st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------------------------
    # PAGE 2: TRANSACTION RISK EVALUATOR
    # -------------------------------------------------------------
    elif nav == "Transaction Risk Evaluator":
        st.markdown(f"""
        <div class="op-header">
            <div style="display: flex; align-items: center; gap: 14px;">
                <img src="data:image/jpeg;base64,{LOGO_B64}" style="width: 46px; height: 46px; object-fit: contain; border-radius: 8px; border: 1px solid #E2E8F0; background: #FFFFFF; padding: 2px;" alt="FinShield" />
                <div>
                    <div class="op-breadcrumb">FinShield / Real-Time Transaction Evaluation</div>
                    <h1 class="op-title">Transaction Risk Evaluator</h1>
                    <p class="op-subtitle">Evaluate incoming transaction parameters against empirical models with SHAP feature attribution.</p>
                </div>
            </div>
            <div class="op-status-badge">
                Operator: {user['full_name']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Tactical Scenario Presets
        st.markdown("<div style='font-size: 0.82rem; font-weight: 600; color: #475569; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 8px;'>TEST SCENARIOS (1-CLICK FILL)</div>", unsafe_allow_html=True)
        p_col1, p_col2, p_col3, p_col4 = st.columns(4)
        
        if "sim_amount" not in st.session_state:
            st.session_state.sim_amount = 250.0
            st.session_state.sim_balance = 4500.0
            st.session_state.sim_age = 35
            st.session_state.sim_occupation = "Engineer"
            st.session_state.sim_channel = "Online"
            st.session_state.sim_location = "Houston"
            st.session_state.sim_duration = 65
            st.session_state.sim_logins = 1
            st.session_state.sim_hour = 14
            st.session_state.sim_time_prev = 120.0
            st.session_state.sim_device = False
            st.session_state.sim_ip = False
            st.session_state.sim_day = "Monday"

        if p_col1.button("Account Takeover (ATO)", use_container_width=True):
            st.session_state.sim_amount = 1450.0
            st.session_state.sim_balance = 2000.0
            st.session_state.sim_age = 42
            st.session_state.sim_occupation = "Doctor"
            st.session_state.sim_channel = "Online"
            st.session_state.sim_location = "Miami"
            st.session_state.sim_duration = 18
            st.session_state.sim_logins = 5
            st.session_state.sim_hour = 3
            st.session_state.sim_time_prev = 10.0
            st.session_state.sim_device = True
            st.session_state.sim_ip = True
            st.session_state.sim_day = "Sunday"
            st.rerun()

        if p_col2.button("Midnight Balance Drain", use_container_width=True):
            st.session_state.sim_amount = 4850.0
            st.session_state.sim_balance = 5000.0
            st.session_state.sim_age = 65
            st.session_state.sim_occupation = "Retired"
            st.session_state.sim_channel = "ATM"
            st.session_state.sim_location = "Las Vegas"
            st.session_state.sim_duration = 35
            st.session_state.sim_logins = 2
            st.session_state.sim_hour = 2
            st.session_state.sim_time_prev = 45.0
            st.session_state.sim_device = True
            st.session_state.sim_ip = False
            st.session_state.sim_day = "Saturday"
            st.rerun()

        if p_col3.button("High-Velocity Carding", use_container_width=True):
            st.session_state.sim_amount = 890.0
            st.session_state.sim_balance = 3200.0
            st.session_state.sim_age = 22
            st.session_state.sim_occupation = "Student"
            st.session_state.sim_channel = "Online"
            st.session_state.sim_location = "New York"
            st.session_state.sim_duration = 12
            st.session_state.sim_logins = 1
            st.session_state.sim_hour = 23
            st.session_state.sim_time_prev = 0.5
            st.session_state.sim_device = False
            st.session_state.sim_ip = True
            st.session_state.sim_day = "Friday"
            st.rerun()

        if p_col4.button("Normal Purchase", use_container_width=True):
            st.session_state.sim_amount = 35.50
            st.session_state.sim_balance = 6200.0
            st.session_state.sim_age = 35
            st.session_state.sim_occupation = "Engineer"
            st.session_state.sim_channel = "Branch"
            st.session_state.sim_location = "Houston"
            st.session_state.sim_duration = 75
            st.session_state.sim_logins = 1
            st.session_state.sim_hour = 12
            st.session_state.sim_time_prev = 720.0
            st.session_state.sim_device = False
            st.session_state.sim_ip = False
            st.session_state.sim_day = "Wednesday"
            st.rerun()

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

        with st.form("interactive_inspection_form"):
            f_col1, f_col2, f_col3 = st.columns(3)
            with f_col1:
                st.markdown("<div style='font-size: 0.82rem; font-weight: 700; color: #2563EB; text-transform: uppercase;'>Financial Parameters</div>", unsafe_allow_html=True)
                amount = st.number_input("Transaction Amount ($)", min_value=1.0, max_value=50000.0, value=float(st.session_state.sim_amount), step=10.0)
                balance = st.number_input("Account Balance ($)", min_value=0.0, max_value=1000000.0, value=float(st.session_state.sim_balance), step=100.0)
                tx_type = st.selectbox("Transaction Type", metadata.get("unique_types", ["Debit", "Credit"]))
                channel = st.selectbox("Channel", metadata.get("unique_channels", ["Online", "ATM", "Branch"]), index=metadata.get("unique_channels", ["Online"]).index(st.session_state.sim_channel) if st.session_state.sim_channel in metadata.get("unique_channels", []) else 0)

            with f_col2:
                st.markdown("<div style='font-size: 0.82rem; font-weight: 700; color: #2563EB; text-transform: uppercase;'>Customer Profile</div>", unsafe_allow_html=True)
                age = st.slider("Customer Age", 18, 90, int(st.session_state.sim_age))
                occ_list = metadata.get("unique_occupations", ["Engineer", "Doctor", "Student", "Retired"])
                occupation = st.selectbox("Customer Occupation", occ_list, index=occ_list.index(st.session_state.sim_occupation) if st.session_state.sim_occupation in occ_list else 0)
                loc_list = metadata.get("unique_locations", ["Houston", "San Diego", "Raleigh"])
                location = st.selectbox("Location", loc_list, index=loc_list.index(st.session_state.sim_location) if st.session_state.sim_location in loc_list else 0)
                login_attempts = st.number_input("Login Attempts", min_value=1, max_value=10, value=int(st.session_state.sim_logins), step=1)

            with f_col3:
                st.markdown("<div style='font-size: 0.82rem; font-weight: 700; color: #2563EB; text-transform: uppercase;'>Security Telemetry</div>", unsafe_allow_html=True)
                duration = st.number_input("Duration (Seconds)", min_value=1, max_value=1200, value=int(st.session_state.sim_duration), step=5)
                tx_hour = st.slider("Hour of Day (0-23)", 0, 23, int(st.session_state.sim_hour))
                day_list = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                day_name = st.selectbox("Day of Week", day_list, index=day_list.index(st.session_state.sim_day))
                time_since_prev = st.number_input("Minutes Since Previous TX", min_value=0.0, max_value=10000.0, value=float(st.session_state.sim_time_prev), step=10.0)
                device_changed = st.checkbox("Device Changed / Unrecognized", value=bool(st.session_state.sim_device))
                ip_changed = st.checkbox("IP Address Changed / Foreign", value=bool(st.session_state.sim_ip))

            submit_eval = st.form_submit_button("Evaluate Transaction Risk", use_container_width=True)

        if submit_eval or True:
            is_weekend = day_name in ["Saturday", "Sunday"]
            amt_balance_ratio = amount / (balance if balance > 0 else 0.01)
            high_val_flag = amount > metadata['upper_limit']

            input_data = pd.DataFrame([{
                'transaction_amount': amount,
                'customer_age': age,
                'transaction_seconds_duration': duration,
                'login_attempts': login_attempts,
                'account_balance': balance,
                'transaction_hour': tx_hour,
                'time_since_previous': time_since_prev,
                'amount_balance_ratio': amt_balance_ratio,
                'transaction_type': tx_type,
                'location': location,
                'channel': channel,
                'customer_occupation': occupation,
                'device_changed': device_changed,
                'ip_changed': ip_changed,
                'is_weekend': is_weekend,
                'high_value_transaction': high_val_flag,
                'day_of_week': day_name
            }])

            X_single = preprocess_for_inference(input_data, scaler, metadata)

            iso_pred = iso_model.predict(X_single)[0]
            lof_pred = lof_model.predict(X_single)[0]
            iso_score = float(iso_model.decision_function(X_single)[0])
            lof_score = float(lof_model.score_samples(X_single)[0])
            
            risk_score = compute_unified_risk_score(iso_score, lof_score)

            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
            gauge_col, info_col = st.columns([1, 1.2])

            with gauge_col:
                st.plotly_chart(create_sleek_risk_gauge(risk_score), use_container_width=True)

            with info_col:
                if risk_score >= 75:
                    st.markdown(f"""
                    <div style='background: #FFF1F2; border: 1px solid #FECDD3; border-radius: 8px; padding: 16px; margin-bottom: 12px;'>
                        <div style='color: #E11D48; font-size: 1.1rem; font-weight: 700;'>CRITICAL FRAUD ANOMALY (Risk: {risk_score}%)</div>
                        <div style='color: #475569; font-size: 0.85rem; margin-top: 4px;'>Severe deviations detected across account balance ratio, hardware profile, or network telemetry. Immediate investigation recommended.</div>
                    </div>
                    """, unsafe_allow_html=True)
                elif risk_score >= 45:
                    st.markdown(f"""
                    <div style='background: #FFFBEB; border: 1px solid #FDE68A; border-radius: 8px; padding: 16px; margin-bottom: 12px;'>
                        <div style='color: #D97706; font-size: 1.1rem; font-weight: 700;'>ELEVATED RISK FLAG (Risk: {risk_score}%)</div>
                        <div style='color: #475569; font-size: 0.85rem; margin-top: 4px;'>Moderate behavioral divergence detected. Additional identity challenge or step-up authentication advised.</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style='background: #ECFDF5; border: 1px solid #A7F3D0; border-radius: 8px; padding: 16px; margin-bottom: 12px;'>
                        <div style='color: #059669; font-size: 1.1rem; font-weight: 700;'>NORMAL TRANSACTION (Risk: {risk_score}%)</div>
                        <div style='color: #475569; font-size: 0.85rem; margin-top: 4px;'>Transaction conforms to the historical account behavioral baseline. Approved without challenge.</div>
                    </div>
                    """, unsafe_allow_html=True)

                sub_c1, sub_c2 = st.columns(2)
                sub_c1.metric("Isolation Forest", "Flagged" if iso_pred == -1 else "Normal", delta=f"{iso_score:.3f}")
                sub_c2.metric("Local Outlier Factor", "Flagged" if lof_pred == -1 else "Normal", delta=f"{lof_score:.3f}")

                # Quick escalate to Case Database
                with st.expander("Escalate to Case Management Queue", expanded=False):
                    analyst_name = st.text_input("Investigator Identity", value=user['full_name'])
                    case_notes = st.text_area("Investigation Notes", value=f"Evaluated risk score of {risk_score}%. Escalated for manual ledger audit.")
                    if st.button("Register Investigation Case", use_container_width=True):
                        tx_id = f"TX-SIM-{np.random.randint(100000, 999999)}"
                        add_case(tx_id, "SIM-ACCOUNT", analyst_name, "Open - Under Investigation", risk_score, "Flagged by Ensemble", case_notes)
                        st.success(f"Case {tx_id} registered successfully in the Case Store.")

            # SHAP Explainability
            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            st.markdown("<div style='font-size: 1.05rem; font-weight: 700; color: #0F172A; margin-bottom: 2px;'>Feature Attribution (SHAP Explainability)</div>", unsafe_allow_html=True)
            st.caption("Quantifies the exact contribution of each parameter toward pushing the decision toward or away from the anomaly threshold.")
            
            shap_df = explain_transaction_shap(explainer, X_single, metadata['feature_columns'], top_n=8)
            
            fig_shap = px.bar(
                shap_df,
                x='Impact',
                y='Feature',
                orientation='h',
                color='Impact',
                color_continuous_scale=[[0, '#EF4444'], [0.5, '#F59E0B'], [1, '#10B981']]
            )
            apply_clean_theme(fig_shap, "Top Parameters Influencing Model Score (Red = Pushing Toward Anomaly)")
            fig_shap.update_layout(yaxis={'categoryorder': 'total ascending'}, height=340)
            st.plotly_chart(fig_shap, use_container_width=True)

    # -------------------------------------------------------------
    # PAGE 3: UNIVERSAL DATASET AUDITOR
    # -------------------------------------------------------------
    elif nav == "Universal Dataset Auditor":
        st.markdown(f"""
        <div class="op-header">
            <div style="display: flex; align-items: center; gap: 14px;">
                <img src="data:image/jpeg;base64,{LOGO_B64}" style="width: 46px; height: 46px; object-fit: contain; border-radius: 8px; border: 1px solid #E2E8F0; background: #FFFFFF; padding: 2px;" alt="FinShield" />
                <div>
                    <div class="op-breadcrumb">FinShield / Batch CSV Processing</div>
                    <h1 class="op-title">Universal Dataset Auditor</h1>
                    <p class="op-subtitle">Upload and audit any banking or transaction CSV via automatic column mapping or instant fresh model training.</p>
                </div>
            </div>
            <div class="op-status-badge">
                Operator: {user['full_name']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        upload_file = st.file_uploader("Upload Transaction Dataset (CSV)", type=["csv"])
        use_baseline_sample = st.checkbox("Or test with baseline Bank Transactions dataset (First 250 records)", value=(upload_file is None))

        custom_df = None
        if upload_file is not None:
            try:
                custom_df = pd.read_csv(upload_file)
                st.success(f"Loaded {len(custom_df):,} rows and {len(custom_df.columns)} columns.")
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
        elif use_baseline_sample and baseline_df is not None:
            custom_df = baseline_df.head(250).copy()
            st.info("Loaded 250 records from baseline dataset.")

        if custom_df is not None:
            st.markdown("<div style='font-size: 0.88rem; font-weight: 600; color: #0F172A; margin: 8px 0;'>Select Processing Engine:</div>", unsafe_allow_html=True)
            audit_mode = st.radio(
                "Select Processing Engine:",
                [
                    "Option A: Smart Column Auto-Mapping & Pretrained Pipeline",
                    "Option B: Fresh Auto-Training directly on this dataset schema"
                ],
                index=0,
                label_visibility="collapsed"
            )

            if "Smart Column Auto-Mapping" in audit_mode:
                detected_mapping = auto_detect_and_map_columns(custom_df)
                
                with st.expander("View Auto-Detected Column Synonyms", expanded=False):
                    st.write("Mapped columns:")
                    st.json(detected_mapping)
                    st.caption("Missing parameters are automatically imputed with conservative statistical defaults.")

                if st.button("Run Pretrained Pipeline Audit", use_container_width=True):
                    with st.spinner("Processing feature engineering & scoring batch..."):
                        df_mapped = custom_df.rename(columns=detected_mapping)
                        enriched = engineer_features(df_mapped, upper_limit=metadata['upper_limit'])
                        X_audit = preprocess_for_inference(enriched, scaler, metadata)

                        iso_scores = iso_model.decision_function(X_audit)
                        lof_scores = lof_model.score_samples(X_audit)

                        threshold = np.percentile(iso_scores, sensitivity)
                        iso_preds = np.where(iso_scores < threshold, -1, 1)
                        lof_preds = np.where(lof_scores < np.percentile(lof_scores, sensitivity), -1, 1)

                        risk_scores = [compute_unified_risk_score(s, l) for s, l in zip(iso_scores, lof_scores)]
                        enriched['Risk_Score_Pct'] = risk_scores
                        enriched['Isolation_Forest'] = np.where(iso_preds == -1, 'Anomaly', 'Normal')
                        enriched['LOF'] = np.where(lof_preds == -1, 'Anomaly', 'Normal')

                        conditions = [
                            (enriched['Isolation_Forest'] == 'Anomaly') & (enriched['LOF'] == 'Anomaly'),
                            (enriched['Isolation_Forest'] == 'Anomaly') | (enriched['LOF'] == 'Anomaly')
                        ]
                        enriched['Audit_Verdict'] = np.select(conditions, ['CRITICAL (Both)', 'WARNING (Single Model)'], default='NORMAL')

                        st.session_state['audit_results'] = enriched

            else:
                st.info("Fits fresh Isolation Forest + LOF models on all numeric and categorical features found in your uploaded file.")
                if st.button("Train Fresh Models & Score Dataset", use_container_width=True):
                    with st.spinner("Fitting unsupervised anomaly detectors on custom dataset..."):
                        df_res, feats = train_fresh_model_on_any_dataset(custom_df, contamination=sensitivity/100.0)
                        st.success(f"Models fitted on {len(feats)} features ({len(df_res)} records).")
                        st.session_state['audit_results'] = df_res

            if 'audit_results' in st.session_state:
                res_df = st.session_state['audit_results']
                st.markdown("---")

                c_total = len(res_df)
                c_crit = (res_df['Audit_Verdict'] == 'CRITICAL (Both)').sum()
                c_warn = (res_df['Audit_Verdict'] == 'WARNING (Single Model)').sum()
                c_norm = (res_df['Audit_Verdict'] == 'NORMAL').sum()

                b1, b2, b3, b4 = st.columns(4)
                b1.metric("Total Scanned", f"{c_total:,}")
                b2.metric("Critical Consensus", f"{c_crit}")
                b3.metric("Single Model Flag", f"{c_warn}")
                b4.metric("Normal Transactions", f"{c_norm}")

                filter_choice = st.selectbox(
                    "Filter View",
                    ["All Records", "CRITICAL (Both Flagged)", "WARNING (Single Model)", "All Flagged Only"]
                )

                if filter_choice == "CRITICAL (Both Flagged)":
                    filtered_df = res_df[res_df['Audit_Verdict'] == 'CRITICAL (Both)']
                elif filter_choice == "WARNING (Single Model)":
                    filtered_df = res_df[res_df['Audit_Verdict'] == 'WARNING (Single Model)']
                elif filter_choice == "All Flagged Only":
                    filtered_df = res_df[res_df['Audit_Verdict'] != 'NORMAL']
                else:
                    filtered_df = res_df

                st.dataframe(filtered_df, use_container_width=True, hide_index=True)

                csv_data = filtered_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download Audit Report (CSV)",
                    data=csv_data,
                    file_name="anomaly_audit_report.csv",
                    mime="text/csv",
                    use_container_width=True
                )

    # -------------------------------------------------------------
    # PAGE 4: FRAUD CASE MANAGEMENT QUEUE
    # -------------------------------------------------------------
    elif nav == "Fraud Case Management Queue":
        st.markdown(f"""
        <div class="op-header">
            <div style="display: flex; align-items: center; gap: 14px;">
                <img src="data:image/jpeg;base64,{LOGO_B64}" style="width: 46px; height: 46px; object-fit: contain; border-radius: 8px; border: 1px solid #E2E8F0; background: #FFFFFF; padding: 2px;" alt="FinShield" />
                <div>
                    <div class="op-breadcrumb">FinShield / Investigation Case Queue</div>
                    <h1 class="op-title">Fraud Investigation Case Store</h1>
                    <p class="op-subtitle">Manage escalated transactions, assign investigator verdicts, and log investigation closures.</p>
                </div>
            </div>
            <div class="op-status-badge">
                Operator: {user['full_name']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        cases_df = get_all_cases()

        st.markdown("<div style='font-size: 0.95rem; font-weight: 600; color: #0F172A; margin-bottom: 8px;'>Active Investigation Queue</div>", unsafe_allow_html=True)
        if not cases_df.empty:
            st.dataframe(cases_df, use_container_width=True, hide_index=True)

            st.markdown("---")
            st.markdown("<div style='font-size: 0.95rem; font-weight: 600; color: #0F172A; margin-bottom: 8px;'>Update Investigation Status</div>", unsafe_allow_html=True)
            u_col1, u_col2, u_col3 = st.columns([1, 1, 2])
            with u_col1:
                case_id = st.selectbox("Select Case ID", cases_df['id'].tolist())
            with u_col2:
                new_status = st.selectbox("Status", ["Open - Under Investigation", "Confirmed Fraud", "False Positive", "Closed"])
            with u_col3:
                new_notes = st.text_input("Investigator Notes", value="Investigation concluded.")

            if st.button("Save Case Updates", use_container_width=True):
                update_case(case_id, new_status, new_notes)
                st.success(f"Case #{case_id} updated successfully.")
                st.rerun()

            csv_cases = cases_df.to_csv(index=False).encode('utf-8')
            st.download_button("Export Investigation Log (CSV)", csv_cases, "fraud_investigation_cases.csv", "text/csv")
        else:
            st.info("No active investigation cases logged yet. Escalate transactions from the Evaluator or register one manually below.")

        st.markdown("---")
        st.markdown("<div style='font-size: 0.95rem; font-weight: 600; color: #0F172A; margin-bottom: 8px;'>Register Manual Investigation Docket</div>", unsafe_allow_html=True)
        with st.form("manual_case_form"):
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                m_tx_id = st.text_input("Transaction ID", value="TX-MANUAL-001")
                m_acc_id = st.text_input("Account ID", value="AC-0099")
            with m_col2:
                m_analyst = st.text_input("Investigator Name", value=user['full_name'])
                m_score = st.slider("Observed Risk Score (%)", 0, 100, 85)
            with m_col3:
                m_verdict = st.selectbox("Assessment", ["Suspicious Velocity", "Device Spoofing", "Account Drain Attempt", "Other"])
                m_notes = st.text_area("Initial Notes", value="Flagged during manual ledger review.")

            if st.form_submit_button("Add Case to Database", use_container_width=True):
                add_case(m_tx_id, m_acc_id, m_analyst, "Open - Under Investigation", m_score, m_verdict, m_notes)
                st.success(f"Case {m_tx_id} successfully registered.")
                st.rerun()

    # -------------------------------------------------------------
    # PAGE 5: MODEL GOVERNANCE & API
    # -------------------------------------------------------------
    elif nav == "Model Governance & API":
        st.markdown(f"""
        <div class="op-header">
            <div style="display: flex; align-items: center; gap: 14px;">
                <img src="data:image/jpeg;base64,{LOGO_B64}" style="width: 46px; height: 46px; object-fit: contain; border-radius: 8px; border: 1px solid #E2E8F0; background: #FFFFFF; padding: 2px;" alt="FinShield" />
                <div>
                    <div class="op-breadcrumb">FinShield / Microservice Architecture</div>
                    <h1 class="op-title">Model Governance &amp; Production REST API</h1>
                    <p class="op-subtitle">Technical architecture, hyperparameters, and enterprise microservice integration specifications.</p>
                </div>
            </div>
            <div class="op-status-badge">
                Operator: {user['full_name']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        d_col1, d_col2 = st.columns(2)
        with d_col1:
            st.markdown("""
            ### Isolation Forest Architecture
            - **Number of Estimators**: `100` trees
            - **Contamination Rate**: `5%` baseline (dynamically tunable via sidebar)
            - **Random Seed**: `42`
            - **Operating Principle**: Partitions feature space using random hyperplanes. Outlier anomalies isolate near the root of the trees with fewer splits.
            """)

        with d_col2:
            st.markdown("""
            ### Local Outlier Factor (LOF)
            - **Nearest Neighbors ($k$)**: `20`
            - **Contamination Rate**: `5%` baseline
            - **Inference Mode**: `Novelty Detection (Fitted baseline)`
            - **Operating Principle**: Evaluates the local density deviation of a sample relative to its nearest surrounding cluster neighbors.
            """)

        st.markdown("---")
        st.markdown("### REST API Integration (`api.py`)")
        st.markdown("""
        Includes a high-performance **FastAPI** backend for core banking and payment processing microservices to query transaction risk in real time.
        
        - **Endpoint**: `http://localhost:8000/api/v1/predict`
        - **Swagger Documentation**: `http://localhost:8000/docs`
        """)

        st.markdown("#### Sample Python Query")
        st.code("""
import requests

payload = {
    "transaction_id": "TX99104",
    "account_id": "AC00452",
    "transaction_amount": 2850.0,
    "account_balance": 3000.0,
    "customer_age": 35,
    "customer_occupation": "Doctor",
    "channel": "Online",
    "location": "Miami",
    "login_attempts": 4,
    "device_changed": True,
    "ip_changed": True,
    "transaction_hour": 3
}

response = requests.post("http://localhost:8000/api/v1/predict", json=payload)
print(response.json())
        """, language="python")

        st.markdown("#### Sample Response")
        st.code("""
{
  "transaction_id": "TX99104",
  "verdict": "CRITICAL_ANOMALY",
  "risk_level": "CRITICAL",
  "risk_score_pct": 91.4,
  "isolation_forest_anomaly": true,
  "lof_anomaly": true,
  "isolation_forest_score": -0.1842,
  "risk_factors": [
    "Amount ($2,850.00) exceeds dataset IQR threshold ($913.49).",
    "High amount-to-balance ratio (95.0%).",
    "4 failed login attempts prior to transaction.",
    "Transaction originates from an unrecognized device.",
    "Transaction originates from an anomalous IP address."
  ]
}
        """, language="json")
