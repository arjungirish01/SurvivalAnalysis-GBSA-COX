import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from patients_db import generate_patient_records

st.set_page_config(
    page_title="OncoCast | Clinical Decision Support",
    layout="wide",
    page_icon="⚕️",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* MAIN BACKGROUND - Soft Clinical Gray */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* --- NEW BANNER DESIGN --- */
    .medical-banner {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        padding: 20px 30px;
        border-radius: 0 0 20px 20px;
        margin-top: -60px; /* Pulls banner to top edge */
        margin-bottom: 30px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.15);
        color: white;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    .banner-left {
        display: flex;
        align-items: center;
        gap: 20px;
    }
    
    .banner-icon {
        font-size: 3.5rem;
        background: linear-gradient(135deg, #00b09b, #96c93d); /* Snake Green Gradient */
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        filter: drop-shadow(0 2px 4px rgba(0,0,0,0.3));
    }
    
    .banner-text h1 {
        margin: 0;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
        font-size: 2.2rem;
        color: #ffffff;
        letter-spacing: 1px;
    }
    
    .banner-text p {
        margin: 0;
        color: #bdc3c7;
        font-size: 0.95rem;
        font-weight: 300;
        letter-spacing: 0.5px;
    }
    
    .banner-right {
        text-align: right;
        display: none; /* Hidden on small screens */
    }
    
    @media (min-width: 800px) {
        .banner-right { display: block; }
    }
    
    .hospital-badge {
        background-color: rgba(255,255,255,0.1);
        border: 1px solid rgba(255,255,255,0.2);
        padding: 5px 15px;
        border-radius: 30px;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #ecf0f1;
    }

    /* --- PATIENT CARD --- */
    .patient-card {
        background-color: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border-top: 5px solid #3498db;
    }
    .patient-name {
        color: #2c3e50;
        font-size: 26px;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .patient-meta {
        color: #7f8c8d;
        font-size: 15px;
        margin-bottom: 5px;
    }
    .status-badge {
        display: inline-block;
        background-color: #e8f8f5;
        color: #27ae60;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 13px;
        font-weight: 600;
        margin-top: 10px;
    }

    /* --- METRICS --- */
    .metric-box {
        background: white;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #edf2f7;
        text-align: center;
        box-shadow: 0 2px 5px rgba(0,0,0,0.03);
    }
</style>
""", unsafe_allow_html=True)

#LOGIC & LOADER

@st.cache_data
def get_data():
    return generate_patient_records(5)

@st.cache_resource
def load_resources():
    try:
        data = joblib.load('./saved_models/gbsa_model.pkl')
        return data
    except FileNotFoundError:
        return None

def preprocess_input(patient_series, trained_columns):
    df = pd.DataFrame([patient_series])
    
    #Binary Encoding
    binary_cols = [
        'psych_disturb', 'diabetes', 'arrhythmia', 'vent_hist', 'renal_issue',
        'pulm_severe', 'rituximab', 'obesity', 'in_vivo_tcd', 'hepatic_severe',
        'prior_tumor', 'peptic_ulcer', 'rheum_issue', 'hepatic_mild', 'cardiac',
        'pulm_moderate', 'mrd_hct'
    ]
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: 1 if str(x).lower() == 'yes' else 0)

    #Imputation Defaults
    defaults = {'donor_age': 30.0, 'karnofsky_score': 90.0, 'comorbidity_score': 0.0, 'year_hct': 2018.0}
    for col, val in defaults.items():
        if col in df.columns: df[col] = df[col].fillna(val)

    #One-Hot Encoding
    cat_cols = [c for c in df.columns if df[c].dtype == 'object' and c not in binary_cols + ['first_name', 'last_name', 'picture', 'ID']]
    df = pd.get_dummies(df, columns=cat_cols)
    
    #Alignment
    model_input = pd.DataFrame(0.0, index=[0], columns=trained_columns)
    common_cols = list(set(df.columns) & set(trained_columns))
    model_input[common_cols] = df[common_cols]
    
    #Safety Net
    return model_input.fillna(0.0)

#UI LAYOUT
st.markdown("""
<div class="medical-banner">
    <div class="banner-left">
        <div class="banner-icon">⚕️</div>
        <div class="banner-text">
            <h1>OncoCast</h1>
            <p>HCT SURVIVAL ANALYTICS SYSTEM</p>
        </div>
    </div>
    <div class="banner-right">
        <div class="hospital-badge">Department of Hematology</div>
        <div style="margin-top:5px; font-size:0.8rem; opacity:0.8;">Clinician ID: DR-8842</div>
    </div>
</div>
""", unsafe_allow_html=True)

#Sidebar
with st.sidebar:
    st.markdown("### Patient Registry")
    patients_df = get_data()
    
    s_fname = st.text_input("First Name")
    s_lname = st.text_input("Last Name")
    s_age = st.number_input("Age", 0, 100)
    
    st.markdown("<br>", unsafe_allow_html=True)
    search_btn = st.button("Retrieve Record", type="primary", use_container_width=True)
    
    st.divider()
    st.caption("v2.1 | Powered by GBSA Model")
    st.caption("Secure Connection: TLS 1.3")
#Search Logic
if search_btn:
    mask = (
        (patients_df['first_name'].str.lower() == s_fname.lower()) &
        (patients_df['last_name'].str.lower() == s_lname.lower()) &
        (patients_df['age'] == s_age)
    )
    results = patients_df[mask]
    
    if not results.empty:
        st.session_state['patient'] = results.iloc[0]
    else:
        st.error("Patient not found in EMR database.")

#Main Display
if 'patient' in st.session_state:
    p = st.session_state['patient']
    
    #PATIENT CARD
    col_pic, col_details = st.columns([1, 4])
    
    with col_pic:
        st.markdown(f"""
        <div style="padding:10px;">
            <img src="{p["picture"]}" style="border-radius:50%; width:140px; height:140px; object-fit:cover; border: 4px solid #fff; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
        </div>
        """, unsafe_allow_html=True)
        
    with col_details:
        st.markdown(f"""
        <div class="patient-card">
            <div class="patient-name">{p['last_name']}, {p['first_name']}</div>
            <div class="patient-meta"><strong>MRN:</strong> {p['ID']} &nbsp;|&nbsp; <strong>DOB:</strong> 19XX (Age {p['age']})</div>
            <div class="patient-meta"><strong>Diagnosis:</strong> {p['prim_disease_hct']} &nbsp;|&nbsp; <strong>Source:</strong> {p['graft_type']}</div>
            <div class="status-badge">● Active Monitoring Phase</div>
        </div>
        """, unsafe_allow_html=True)

    #EVALUATION
    st.write("")
    if st.button("Run Survival Risk Model", type="primary", use_container_width=True):
        bundle = load_resources()
        
        if bundle:
            if isinstance(bundle, dict):
                model, cols = bundle['model'], bundle['features']
            else:
                model = bundle
                cols = model.feature_names_in_.tolist() if hasattr(model, "feature_names_in_") else []
            
            if not cols:
                st.error("Model schema missing.")
                st.stop()
                
            X_input = preprocess_input(p, cols)
            risk_score = model.predict(X_input)[0]
            surv_funcs = model.predict_survival_function(X_input)
            
            #RESULTS
            st.markdown("### Prognostic Analysis")
            st.markdown("<hr style='margin:10px 0;'>", unsafe_allow_html=True)
            
            res_c1, res_c2 = st.columns([2, 1])
            
            with res_c1:
                fn = surv_funcs[0]
                times = fn.x
                probs = fn.y
            
                baseline_probs = np.exp(-0.0008 * times)
                
                fig = go.Figure()

                #Population Baseline
                fig.add_trace(go.Scatter(
                    x=times, y=baseline_probs,
                    mode='lines',
                    name='Population Avg.',
                    line=dict(color='#b2bec3', width=2, dash='dash'),
                    hoverinfo='skip'
                ))

                #Patient Specific Curve
                fig.add_trace(go.Scatter(
                    x=times, y=probs,
                    mode='lines',
                    name=f"{p['last_name']}, {p['first_name']}",
                    line=dict(shape='hv', color='#0984e3', width=3), 
                    fill='tozeroy',
                    fillcolor='rgba(9, 132, 227, 0.1)'
                ))

                #Layout Styling
                fig.update_layout(
                    title="<b>Survival Projection vs. Baseline</b>",
                    title_font=dict(size=16, color='#2d3436'),
                    xaxis_title="Days Post-Transplant",
                    yaxis_title="Survival Probability",
                    yaxis=dict(range=[0, 1.05], gridcolor='#dfe6e9'),
                    xaxis=dict(gridcolor='#dfe6e9'),
                    plot_bgcolor='white',
                    margin=dict(l=20, r=20, t=40, b=20),
                    hovermode="x unified", # Shows all data points at that x-value
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                one_year_idx = (np.abs(times - 365)).argmin()
                one_year_prob = probs[one_year_idx]
                
                fig.add_annotation(
                    x=times[one_year_idx], y=one_year_prob,
                    text=f"1-Year: {one_year_prob:.0%}",
                    showarrow=True, arrowhead=1,
                    ax=0, ay=-40,
                    bgcolor="#2d3436", font=dict(color="white", size=10)
                )

                st.plotly_chart(fig, use_container_width=True)
            
            with res_c2:
                #Hazard Metric
                st.markdown(f"""
                <div class="metric-box">
                    <div style="color:#7f8c8d; font-size:0.9rem; text-transform:uppercase;">Hazard Ratio</div>
                    <div style="color:#2d3436; font-size:2.2rem; font-weight:bold;">{risk_score:.3f}</div>
                    <div style="font-size:0.8rem; color:#b2bec3;">vs Population Baseline</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<h5 style='margin-top:20px; color:#2c3e50;'>Primary Risk Drivers</h5>", unsafe_allow_html=True)
                
                #Factors
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1][:4]
                
                for idx in indices:
                    feat = cols[idx]
                    score = importances[idx]
                    
                    display = feat.replace('_', ' ').title().replace("Score", "")
                    
                    #Remove "Nan", "N/A", "Na" explicitly
                    garbage_terms = [" Nan", " N/A", " Na", "Nan", "N/A"]
                    for term in garbage_terms:
                        display = display.replace(term, "")
                    
                    display = display.strip() # Remove trailing spaces
                    
                    #Value Lookup
                    raw_val = "N/A"
                    key_attempt = feat.split('_')[0]
                    if feat in p: raw_val = p[feat]
                    elif key_attempt in p: raw_val = p[key_attempt]
                    
                    st.markdown(f"""
                    <div style="margin-bottom: 8px;">
                        <div style="display:flex; justify-content:space-between; font-size:14px; font-weight:600; color:#2d3436;">
                            <span>{display}</span>
                            <span style="color:#e17055;">{raw_val}</span>
                        </div>
                        <div style="background-color:#dfe6e9; height:5px; border-radius:3px; margin-top:2px;">
                            <div style="background: linear-gradient(90deg, #74b9ff, #0984e3); width: {min(score*500, 100)}%; height:100%; border-radius:3px;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        else:
            st.error("Model file not found.")

elif not search_btn:
    st.markdown("""
    <div style="text-align: center; padding: 60px; color: #b2bec3;">
        <h3>Waiting for Input</h3>
        <p>Use the sidebar to search for a patient record.</p>
    </div>
    """, unsafe_allow_html=True)