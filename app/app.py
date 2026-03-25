import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os

st.set_page_config(
    page_title="Indoor Thermal Comfort Digital Twin",
    page_icon="🏢",
    layout="wide"
)

# ------------------- CUSTOM CSS -------------------
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Main container background with animated gradient */
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .stApp {
        background: linear-gradient(-45deg, #0f172a, #1e293b, #0c1a30, #162444);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
        color: #f8fafc;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.8) !important;
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    [data-testid="stSidebar"] * {
        color: #f8fafc !important;
    }

    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 30px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        margin-bottom: 24px;
        transition: transform 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275), box-shadow 0.3s ease;
    }
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 45px 0 rgba(0, 0, 0, 0.5);
        border-color: rgba(255, 255, 255, 0.2);
    }

    /* Buttons - Glow and Pulse */
    @keyframes pulse-glow {
        0% { box-shadow: 0 0 0 0 rgba(124, 58, 237, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(124, 58, 237, 0); }
        100% { box-shadow: 0 0 0 0 rgba(124, 58, 237, 0); }
    }

    div.stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        background-size: 200% auto;
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 2rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        text-transform: uppercase;
        animation: pulse-glow 2s infinite;
    }
    div.stButton > button:hover {
        background-position: right center;
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 10px 25px rgba(168, 85, 247, 0.5);
    }

    /* Headers and Dividers */
    h1 {
        font-weight: 800 !important;
        background: -webkit-linear-gradient(45deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
        margin-bottom: 0.5rem !important;
    }
    
    h2, h3 {
        color: #f1f5f9 !important;
        font-weight: 600 !important;
        letter-spacing: -0.5px;
    }
    hr {
        border-color: rgba(255, 255, 255, 0.1) !important;
        margin: 2rem 0 !important;
    }

    /* Prediction Badges - Advanced */
    @keyframes slideUpFade {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .badge-base {
        padding: 16px 24px;
        border-radius: 16px;
        font-size: 1.4rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 24px;
        animation: slideUpFade 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        border: 1px solid rgba(255,255,255,0.2);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .badge-cold {
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.8), rgba(59, 130, 246, 0.9));
        box-shadow: 0 8px 32px rgba(59, 130, 246, 0.5);
        color: #ffffff;
    }
    .badge-warm {
        background: linear-gradient(135deg, rgba(249, 115, 22, 0.8), rgba(239, 68, 68, 0.9));
        box-shadow: 0 8px 32px rgba(239, 68, 68, 0.5);
        color: #ffffff;
    }
    .badge-neutral {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.8), rgba(5, 150, 105, 0.9));
        box-shadow: 0 8px 32px rgba(16, 185, 129, 0.5);
        color: #ffffff;
    }
    
    /* Metrics Override */
    [data-testid="stMetricValue"] {
        color: #c084fc !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 1rem !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Slider Customization */
    .stSlider > div > div > div > div {
        background-color: #818cf8 !important;
    }
    .stSlider > div > div > div > div[role="slider"] {
        background-color: #c084fc !important;
        box-shadow: 0 0 10px rgba(192, 132, 252, 0.8) !important;
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.03) !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        color: #c084fc !important;
    }
</style>
""", unsafe_allow_html=True)

# -------- Page Navigation --------
if "page" not in st.session_state:
    st.session_state.page = "home"


# ------------------- Helper mappings -------------------
def air_velocity_from_option(option):
    return {
        "Still Air (Fan OFF)": 0.1,
        "Fan LOW": 0.25,
        "Fan HIGH": 0.45
    }[option]

def humidity_from_option(option):
    return {
        "Dry": 35.0,
        "Comfortable": 50.0,
        "Humid": 65.0
    }[option]

def clo_from_option(option):
    return {
        "Light (T-shirt)": 0.5,
        "Normal (Office Wear)": 0.8,
        "Heavy (Jacket)": 1.2
    }[option]

def met_from_option(option):
    return {
        "Sitting": 1.0,
        "Office Work": 1.2,
        "Walking": 1.6
    }[option]


# suggestion 
# ------------------- Dynamic Comfort Suggestions -------------------
def get_dynamic_comfort_suggestions(prediction, ta, rh, v, clo, met):
    suggestions = []

    TEMP_HIGH = 28
    TEMP_LOW = 20

    HUMID_HIGH = 60
    AIR_LOW = 0.15
    AIR_HIGH = 0.35

    CLO_HIGH = 1.0
    CLO_LOW = 0.6

    MET_HIGH = 1.4
    MET_LOW = 1.1

    if prediction == "Warm":
        if ta > TEMP_HIGH:
            suggestions.append(f"Reduce air temperature (currently {ta}°C).")
        if rh > HUMID_HIGH:
            suggestions.append(f"High humidity ({rh}%) — use ventilation or dehumidifier.")
        if v < AIR_LOW:
            suggestions.append(f"Low air movement ({v} m/s) — increase fan speed.")
        if clo > CLO_HIGH:
            suggestions.append(f"Heavy clothing (clo={clo}) — wear lighter clothes.")
        if met > MET_HIGH:
            suggestions.append(f"High activity (met={met}) — reduce activity level.")

    elif prediction == "Cold":
        if ta < TEMP_LOW:
            suggestions.append(f"Increase air temperature (currently {ta}°C).")
        if v > AIR_HIGH:
            suggestions.append(f"High airflow ({v} m/s) — reduce fan speed.")
        if clo < CLO_LOW:
            suggestions.append(f"Light clothing (clo={clo}) — wear warmer clothes.")
        if met < MET_LOW:
            suggestions.append(f"Low activity (met={met}) — slight movement helps warmth.")

    else:
        suggestions.append("Indoor conditions are comfortable.")
        suggestions.append("Maintain current environment settings.")

    if not suggestions:
        suggestions.append("Minor adjustments may further improve comfort.")

    return suggestions


# ------------------- Load Model -------------------
import os
import sys

# Add the parent directory to Python path so `src` module can be found when Streamlit runs inside `app/`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import load_raw_data
from src.preprocessing import preprocess_data
from src.model_training import train_models

MODEL_PATH = os.path.join("models", "thermal_comfort_model.pkl")

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    # Train model dynamically (for deployment)
    df = load_raw_data()
    X, y = preprocess_data(df)
    model = train_models(X, y)


# ------------------- HOME / INDEX PAGE -------------------
def show_home_page():

    # Center layout using columns
    left, center, right = st.columns([1, 2, 1])

    with center:

        st.markdown(
            """
            <h1 style='text-align:center;'>🏢 Indoor Thermal Comfort Digital Twin</h1>
            <p style='text-align:center; font-size:18px; color:gray;'>
            AI-driven system to simulate indoor environments and predict human thermal comfort
            </p>
            """,
            unsafe_allow_html=True
        )

        st.divider()

        st.markdown(
            """
            ### 🌍 Project Overview

            This application demonstrates how **Machine Learning and Digital Twin technology**
            can be used to analyze indoor environmental conditions and predict thermal comfort.

            A **Digital Twin** represents a virtual model of the indoor environment where
            different conditions can be simulated without affecting the real system.

            The system predicts how occupants will feel based on environmental and personal parameters.
            """
        )

        st.markdown(
            """
            ### 📊 Parameters Considered

            The prediction model evaluates multiple factors influencing comfort:

            - 🌡 Air Temperature  
            - 💧 Relative Humidity  
            - 🌬 Air Velocity  
            - 🔥 Radiant Temperature  
            - 🧥 Clothing Insulation  
            - 🏃 Metabolic Rate
            """
        )

        st.markdown(
            """
            ### ⚙️ System Features

            - Machine Learning based comfort prediction  
            - Digital Twin indoor environment simulation  
            - Interactive parameter controls  
            - Real-time comfort classification  
            - Confidence analysis for predictions  
            - Personalized comfort improvement suggestions
            """
        )

        st.markdown(
            """
            ### 🔄 System Workflow

            1️⃣ User inputs indoor environmental parameters  
            2️⃣ Digital Twin simulates the indoor environment  
            3️⃣ Machine Learning model predicts thermal comfort  
            4️⃣ System displays comfort level and recommendations
            """
        )

        st.divider()

        if st.button("🚀 Start Thermal Comfort Simulation", use_container_width=True):
            st.session_state.page = "app"
            st.rerun()


# ------------------- MAIN APPLICATION -------------------
def show_main_app():

    if st.button("⬅ Back to Home"):
        st.session_state.page = "home"
        st.rerun()

    # ------------------- HEADER -------------------
    st.markdown(
        """
        <h1 style='text-align:center;'>🏢 AI Driven Indoor Thermal Comfort Prediction Using Digital Twin Concept</h1>
        <p style='text-align:center; color:gray;'>
        AI-based system to simulate and predict occupant thermal comfort under varying indoor conditions
        </p>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    # ------------------- SIDEBAR -------------------
    st.sidebar.title("⚙️ Environment Settings")
    st.sidebar.caption("Human-friendly controls with manual override")

    ta = st.sidebar.slider(
        "🌡️ Air Temperature (°C)",
        min_value=1.0,
        max_value=45.0,
        value=25.0,
        help="Indoor air temperature (1°C to 45°C)"
    )

    humidity_mode = st.sidebar.radio("💧 Humidity Mode", ["Preset", "Manual"])
    rh = humidity_from_option(
        st.sidebar.selectbox("Humidity Level", ["Dry", "Comfortable", "Humid"])
    ) if humidity_mode == "Preset" else st.sidebar.slider("Humidity (%)", 20.0, 90.0, 50.0)

    airflow_mode = st.sidebar.radio("🌬️ Airflow Mode", ["Preset", "Manual"])
    v = air_velocity_from_option(
        st.sidebar.selectbox("Airflow Condition", ["Still Air (Fan OFF)", "Fan LOW", "Fan HIGH"])
    ) if airflow_mode == "Preset" else st.sidebar.slider("Air Velocity (m/s)", 0.05, 0.60, 0.15)

    clothing_mode = st.sidebar.radio("🧥 Clothing Mode", ["Preset", "Manual"])
    clo = clo_from_option(
        st.sidebar.selectbox("Clothing Level", ["Light (T-shirt)", "Normal (Office Wear)", "Heavy (Jacket)"])
    ) if clothing_mode == "Preset" else st.sidebar.slider("Clothing (clo)", 0.3, 1.5, 0.8)

    activity_mode = st.sidebar.radio("🏃 Activity Mode", ["Preset", "Manual"])
    met = met_from_option(
        st.sidebar.selectbox("Activity Level", ["Sitting", "Office Work", "Walking"])
    ) if activity_mode == "Preset" else st.sidebar.slider("Metabolic Rate (met)", 1.0, 2.5, 1.2)

    radiant_mode = st.sidebar.radio("🔥 Radiant Temperature", ["Auto", "Manual"])
    tr = ta if radiant_mode == "Auto" else st.sidebar.slider("Radiant Temp (°C)", 18.0, 35.0, ta)


    # ------------------- INPUT DATA -------------------
    input_df = pd.DataFrame([{
        "Air temperature (C)": ta,
        "Relative humidity (%)": rh,
        "Air velocity (m/s)": v,
        "Radiant temperature (C)": tr,
        "Clo": clo,
        "Met": met
    }])



    # ------------------- MAIN CONTENT -------------------
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("📥 Current Indoor Conditions")
        
        # Display inputs nicely as metrics instead of raw dataframe
        m1, m2 = st.columns(2)
        m1.metric("Air Temperature", f"{ta} °C")
        m2.metric("Relative Humidity", f"{rh} %")
        
        m3, m4 = st.columns(2)
        m3.metric("Air Velocity", f"{v} m/s")
        m4.metric("Radiant Temp", f"{tr} °C")
        
        m5, m6 = st.columns(2)
        m5.metric("Clothing (Clo)", f"{clo}")
        m6.metric("Activity (Met)", f"{met}")
        
        st.markdown("</div>", unsafe_allow_html=True)

        st.info("💡 This panel represents the **virtual indoor environment** being simulated by the Digital Twin.")

    with col2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("🎯 Digital Twin Prediction")

        if st.button("🚀 Run Simulation", use_container_width=True):
            prediction = model.predict(input_df)[0]
            probabilities = model.predict_proba(input_df)[0]
            classes = model.classes_

            suggestions = get_dynamic_comfort_suggestions(
                prediction, ta, rh, v, clo, met
            )

            if prediction == "Cold":
                st.markdown("<div class='badge-base badge-cold'>❄️ COLD ENVIRONMENT</div>", unsafe_allow_html=True)
            elif prediction == "Warm":
                st.markdown("<div class='badge-base badge-warm'>🔥 WARM ENVIRONMENT</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='badge-base badge-neutral'>✅ COMFORTABLE (NEUTRAL)</div>", unsafe_allow_html=True)

            st.markdown(
                f"""
                <p style="color:#cbd5e1; font-size:1.1rem; text-align:center;">
                Based on the current configuration, the predicted thermal comfort state is <strong>{prediction}</strong>.
                </p>
                """, unsafe_allow_html=True
            )
            
            st.markdown("---")

            with st.expander("🛠️ Personalized Comfort Improvement Suggestions", expanded=True):
                if suggestions:
                    for s in suggestions:
                        st.markdown(f"- {s}")
                
                st.markdown("#### 🧠 Context")
                reasons = []
                if ta >= 28: reasons.append("Relatively high indoor air temperature.")
                elif ta <= 20: reasons.append("Low indoor air temperature.")
                if v >= 0.35: reasons.append("High air movement improves cooling effect.")
                elif v <= 0.12: reasons.append("Low air movement reduces cooling.")
                if clo >= 1.0: reasons.append("Higher clothing insulation traps body heat.")
                if met >= 1.4: reasons.append("Higher activity level increases metabolic heat.")

                if reasons:
                    for r in reasons:
                        st.markdown(f"- {r}")
                else:
                    st.markdown("- Indoor parameters are balanced for comfort.")

            st.markdown("---")
            st.subheader("📊 Thermal Comfort Confidence")

            # Updated Pie Chart Styling (Dark mode friendly)
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor('none')
            ax.set_facecolor('none')

            colors = ["#3b82f6", "#10b981", "#ef4444"]  # Cold, Neutral, Warm

            wedges, texts, autotexts = ax.pie(
                probabilities,
                labels=classes,
                autopct="%1.1f%%",
                startangle=90,
                colors=colors,
                explode=[0.08 if cls == prediction else 0 for cls in classes],
                textprops=dict(color="white")
            )
            
            # Make the chart a donut
            centre_circle = plt.Circle((0,0),0.70,fc='#1e293b') # Dark circle in middle
            fig.gca().add_artist(centre_circle)

            ax.axis("equal")
            st.pyplot(fig)
            
        st.markdown("</div>", unsafe_allow_html=True)


    # ------------------- FOOTER -------------------
    st.divider()
    st.caption(
        "Final Year Project | AI-Driven Indoor Thermal Comfort Prediction using Digital Twin Concepts"
    )


# ------------------- PAGE CONTROLLER -------------------
if st.session_state.page == "home":
    show_home_page()

elif st.session_state.page == "app":
    show_main_app()