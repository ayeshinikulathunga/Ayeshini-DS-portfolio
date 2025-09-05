import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Custom CSS for styling with borders
st.markdown("""
<style>
    /* Main container border */
    .main .block-container {
        border: 3px solid #4CAF50;
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        background-color: #f8f9fa;
    }
    
    /* Title styling */
    .main h1 {
        color: #2E7D32;
        text-align: center;
        border-bottom: 2px solid #4CAF50;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    
    /* Mode selector styling */
    .mode-selector {
        background-color: white;
        border: 2px solid #4CAF50;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        text-align: center;
    }
    
    /* Input sections border */
    .stSelectbox, .stNumberInput {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 5px;
        margin: 5px 0;
        background-color: white;
    }
    
    /* Button styling */
    .stButton > button {
        border: 2px solid #4CAF50;
        border-radius: 10px;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        padding: 10px 20px;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        border-color: #45a049;
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Result boxes with enhanced borders */
    .stAlert {
        border-radius: 10px;
        border: 2px solid;
        margin: 20px 0;
        padding: 15px;
    }
    
    /* Custom section dividers */
    .section-divider {
        border-top: 2px solid #e0e0e0;
        margin: 20px 0;
    }
    
    /* Input labels styling */
    .stSelectbox label, .stNumberInput label {
        font-weight: bold;
        color: #333;
    }
    
    /* Help text styling */
    .help-text {
        font-size: 12px;
        color: #666;
        font-style: italic;
        margin-top: 5px;
    }
    
    /* Mode description styling */
    .mode-description {
        background-color: #e8f5e8;
        border-left: 4px solid #4CAF50;
        padding: 10px;
        margin: 10px 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load models (with error handling for demo purposes)
try:
    model = joblib.load(os.path.join(BASE_DIR, "heart_disease_rf_model.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "heart_scaler.pkl"))
    feature_columns = joblib.load(os.path.join(BASE_DIR, "feature_columns.pkl"))
except:
    st.error("Model files not found. Please ensure all model files are in the correct directory.")
    st.stop()

st.title("Heart Disease Risk Assessment Tool")

# Mode Selection

st.markdown("### Choose Your User Mode")
user_mode = st.selectbox(
    "Select the mode that best describes you:",
    options=["Professional Mode", "Patient Mode", "Family Mode"],
    help="Different modes provide information tailored to your background and needs"
)

# Mode descriptions and content based on selection
mode_descriptions = {
    "Professional Mode": {
        "description": " **Professional Mode**: Medical terminology and detailed technical information for healthcare professionals.",
        "intro": "Provide patient clinical data to assess cardiovascular risk using validated parameters."
    },
    "Patient Mode": {
        "description": "**Patient Mode**: Clear, simple explanations to help you understand your heart health assessment.",
        "intro": "Answer questions about your health to get an easy-to-understand assessment of your heart disease risk."
    },
    "Family Mode": {
        "description": " **Family Mode**: Basic explanations perfect for family members seeking to understand a loved one's condition.",
        "intro": "Help assess heart disease risk for your family member with simple, clear guidance."
    }
}

st.write(mode_descriptions[user_mode]["intro"])

# Add a visual divider
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# Define content based on user mode
def get_field_labels_and_help(mode):
    if mode == "Professional Mode":
        return {
            "age": {"label": "Age (years)", "help": "Patient's age in years"},
            "sex": {"label": "Sex", "help": "Biological sex of the patient"},
            "trestbps": {"label": "Resting Blood Pressure (mm Hg)", "help": "Resting blood pressure measurement"},
            "chol": {"label": "Serum Cholesterol (mg/dl)", "help": "Total serum cholesterol level"},
            "thalach": {"label": "Maximum Heart Rate Achieved", "help": "Peak heart rate during stress testing"},
            "cp": {"label": "Chest Pain Type", "help": "Classification of chest pain symptoms"},
            "exang": {"label": "Exercise Induced Angina", "help": "Angina symptoms triggered by physical activity"},
            "oldpeak": {"label": "ST Depression (Oldpeak)", "help": "ST depression induced by exercise relative to rest"},
            "fbs": {"label": "Fasting Blood Sugar > 120 mg/dl", "help": "Elevated fasting glucose indicator"},
            "restecg": {"label": "Resting ECG Results", "help": "Resting electrocardiographic findings"},
            "slope": {"label": "Slope of Peak Exercise ST Segment", "help": "ST segment slope during peak exercise"},
            "ca": {"label": "Number of Major Vessels (0-3) Colored by Fluoroscopy", "help": "Coronary vessels with significant stenosis"},
            "thal": {"label": "Thalassemia", "help": "Thalassemia stress test results"}
        }
    elif mode == "Patient Mode":
        return {
            "age": {"label": "Your Age", "help": "How old are you?"},
            "sex": {"label": "Your Gender", "help": "Are you male or female?"},
            "trestbps": {"label": "Your Blood Pressure (when resting)", "help": "What's your blood pressure when you're sitting calmly? Normal is around 120/80"},
            "chol": {"label": "Your Cholesterol Level", "help": "Your total cholesterol from blood test. Normal is under 200"},
            "thalach": {"label": "Your Maximum Heart Rate During Exercise", "help": "Highest heart rate you can achieve during exercise or stress test"},
            "cp": {"label": "Type of Chest Pain You Experience", "help": "What kind of chest discomfort do you have?"},
            "exang": {"label": "Do You Get Chest Pain During Exercise?", "help": "Does physical activity cause chest discomfort?"},
            "oldpeak": {"label": "Heart Rhythm Changes During Exercise", "help": "Changes in your heart's electrical activity during exercise (from stress test)"},
            "fbs": {"label": "Is Your Blood Sugar High When Fasting?", "help": "Is your blood sugar over 120 when you haven't eaten? (from blood test)"},
            "restecg": {"label": "Your Heart Test Results (EKG/ECG)", "help": "Results from your heart rhythm test while resting"},
            "slope": {"label": "Heart Response to Exercise", "help": "How your heart's electrical activity responds to exercise"},
            "ca": {"label": "Number of Blocked Heart Arteries", "help": "How many major heart arteries show blockages (from special X-ray)"},
            "thal": {"label": "Heart Blood Flow Test Results", "help": "Results from test showing blood flow to heart muscle"}
        }
    else:  # Family Mode
        return {
            "age": {"label": "Patient's Age", "help": "How old is your family member?"},
            "sex": {"label": "Gender", "help": "Is your family member male or female?"},
            "trestbps": {"label": "Blood Pressure (at rest)", "help": "Blood pressure when sitting quietly. Normal is about 120/80"},
            "chol": {"label": "Cholesterol Level", "help": "Total cholesterol from blood work. Good levels are under 200"},
            "thalach": {"label": "Highest Heart Rate During Activity", "help": "The fastest their heart beat during exercise or a stress test"},
            "cp": {"label": "Type of Chest Discomfort", "help": "What kind of chest pain or discomfort do they experience?"},
            "exang": {"label": "Chest Pain When Active?", "help": "Do they get chest pain or discomfort during physical activity?"},
            "oldpeak": {"label": "Heart Changes During Exercise", "help": "Heart rhythm changes during exercise (measured by doctors)"},
            "fbs": {"label": "High Blood Sugar When Fasting?", "help": "Is their blood sugar high when they haven't eaten? (over 120)"},
            "restecg": {"label": "Heart Test Results (EKG)", "help": "Results from the heart rhythm test done while resting"},
            "slope": {"label": "Heart's Response to Exercise", "help": "How their heart responds to physical activity (from stress test)"},
            "ca": {"label": "Number of Blocked Heart Vessels", "help": "How many major heart blood vessels have blockages"},
            "thal": {"label": "Heart Blood Flow Test", "help": "Test results showing how well blood flows to the heart muscle"}
        }

labels_and_help = get_field_labels_and_help(user_mode)

# Create columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.markdown("###  Basic Information")
    age = st.number_input(
        labels_and_help["age"]["label"], 
        min_value=20, max_value=100, value=50,
        help=labels_and_help["age"]["help"]
    )
    sex = st.selectbox(
        labels_and_help["sex"]["label"], 
        options={0: "Female", 1: "Male"}.keys(), 
        format_func=lambda x: "Female" if x==0 else "Male",
        help=labels_and_help["sex"]["help"]
    )
    
    st.markdown("###  Vital Signs")
    trestbps = st.number_input(
        labels_and_help["trestbps"]["label"], 
        min_value=80, max_value=200, value=120,
        help=labels_and_help["trestbps"]["help"]
    )
    chol = st.number_input(
        labels_and_help["chol"]["label"], 
        min_value=100, max_value=600, value=200,
        help=labels_and_help["chol"]["help"]
    )
    thalach = st.number_input(
        labels_and_help["thalach"]["label"], 
        min_value=70, max_value=220, value=150,
        help=labels_and_help["thalach"]["help"]
    )

with col2:
    st.markdown("###  Symptoms & Conditions")
    
    # Chest pain options based on mode
    if user_mode == "Professional Mode":
        cp_options = {0: "Typical Angina", 1: "Atypical Angina", 2: "Non-anginal Pain", 3: "Asymptomatic"}
    else:
        cp_options = {0: "Classic heart pain", 1: "Unusual heart pain", 2: "Non-heart related pain", 3: "No chest pain"}
    
    cp = st.selectbox(
        labels_and_help["cp"]["label"],
        options=cp_options.keys(), 
        format_func=lambda x: cp_options[x],
        help=labels_and_help["cp"]["help"]
    )
    
    exang = st.selectbox(
        labels_and_help["exang"]["label"], 
        options={0: "No", 1: "Yes"}.keys(), 
        format_func=lambda x: "Yes" if x==1 else "No",
        help=labels_and_help["exang"]["help"]
    )
    
    oldpeak = st.number_input(
        labels_and_help["oldpeak"]["label"], 
        min_value=0.0, max_value=10.0, value=1.0, step=0.1,
        help=labels_and_help["oldpeak"]["help"]
    )

# Full width section for medical tests
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown("### Medical Test Results")

col3, col4 = st.columns(2)

with col3:
    fbs = st.selectbox(
        labels_and_help["fbs"]["label"], 
        options={0: "No", 1: "Yes"}.keys(), 
        format_func=lambda x: "Yes" if x==1 else "No",
        help=labels_and_help["fbs"]["help"]
    )
    
    # ECG options based on mode
    if user_mode == "Professional Mode":
        restecg_options = {0: "Normal", 1: "ST-T Wave Abnormality", 2: "Left Ventricular Hypertrophy"}
    else:
        restecg_options = {0: "Normal", 1: "Minor abnormalities", 2: "Heart enlargement"}
    
    restecg = st.selectbox(
        labels_and_help["restecg"]["label"],
        options=restecg_options.keys(), 
        format_func=lambda x: restecg_options[x],
        help=labels_and_help["restecg"]["help"]
    )

with col4:
    # Slope options based on mode
    if user_mode == "Professional Mode":
        slope_options = {0: "Upsloping", 1: "Flat", 2: "Downsloping"}
    else:
        slope_options = {0: "Good response", 1: "Normal response", 2: "Poor response"}
    
    slope = st.selectbox(
        labels_and_help["slope"]["label"],
        options=slope_options.keys(), 
        format_func=lambda x: slope_options[x],
        help=labels_and_help["slope"]["help"]
    )
    
    ca = st.number_input(
        labels_and_help["ca"]["label"], 
        min_value=0, max_value=3, value=0,
        help=labels_and_help["ca"]["help"]
    )

# Thal options based on mode
if user_mode == "Professional Mode":
    thal_options = {0: "Normal", 1: "Fixed Defect", 2: "Reversible Defect"}
else:
    thal_options = {0: "Normal blood flow", 1: "Permanent reduced flow", 2: "Temporary reduced flow"}

thal = st.selectbox(
    labels_and_help["thal"]["label"],
    options=thal_options.keys(), 
    format_func=lambda x: thal_options[x],
    help=labels_and_help["thal"]["help"]
)

# Convert to feature vector
user_data = {col: 0 for col in feature_columns}

user_data["age"] = age
user_data["sex"] = sex
user_data["trestbps"] = trestbps
user_data["chol"] = chol
user_data["fbs"] = fbs
user_data["thalach"] = thalach
user_data["exang"] = exang
user_data["oldpeak"] = oldpeak
user_data["ca"] = ca

user_data[f"cp_{cp}"] = 1
user_data[f"restecg_{restecg}"] = 1
user_data[f"slope_{slope}"] = 1
user_data[f"thal_{thal}"] = 1

# Center the predict button
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])

with col_btn2:
    if st.button("🔍 Assess Heart Disease Risk", use_container_width=True):
        input_df = pd.DataFrame([user_data])[feature_columns]
        input_scaled = scaler.transform(input_df)
        
        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]
        
        # Results based on user mode
        if user_mode == "Professional Mode":
            if pred == 1:
                st.error(f"⚠️ **High cardiovascular risk detected** \n\n**Risk Probability: {prob*100:.2f}%**\n\nRecommend immediate cardiology consultation and further diagnostic workup.")
            else:
                st.success(f"✅ **Low cardiovascular risk indicated** \n\n**Risk Probability: {prob*100:.2f}%**\n\nContinue standard preventive care and lifestyle modifications.")
        
        elif user_mode == "Patient Mode":
            if pred == 1:
                st.error(f"⚠️ **Higher risk for heart disease detected** \n\n**Your risk level: {prob*100:.2f}%**\n\n**What this means:** The assessment suggests you may be at higher risk for heart problems. This doesn't mean you definitely have heart disease, but it's important to talk with your doctor soon.\n\n**Next steps:** Please schedule an appointment with your healthcare provider to discuss these results and plan next steps for your heart health.")
            else:
                st.success(f"✅ **Lower risk for heart disease** \n\n**Your risk level: {prob*100:.2f}%**\n\n**What this means:** Based on the information provided, you appear to have a lower risk for heart disease. This is good news!\n\n**Keep it up:** Continue with healthy lifestyle choices like regular exercise, good diet, and regular check-ups with your doctor.")
        
        else:  # Family Mode
            if pred == 1:
                st.error(f"⚠️ **Higher heart disease risk found** \n\n**Risk level: {prob*100:.2f}%**\n\n**What this means for your family member:** The assessment suggests they may be at higher risk for heart problems. This is a warning sign that shouldn't be ignored.\n\n**What you can do:** Help them schedule a doctor's appointment soon. Offer support with healthy lifestyle changes like eating better, staying active, and taking medications as prescribed.\n\n**Remember:** This is just a screening tool. Only a doctor can provide a proper diagnosis and treatment plan.")
            else:
                st.success(f"✅ **Lower heart disease risk** \n\n**Risk level: {prob*100:.2f}%**\n\n**Good news for your family member:** Based on this assessment, they appear to have a lower risk for heart disease.\n\n**Keep supporting them:** Encourage them to maintain healthy habits like regular exercise, eating well, and seeing their doctor for regular check-ups.\n\n**Stay vigilant:** Even with lower risk, it's important to watch for any concerning symptoms and maintain regular healthcare visits.")

# Add footer with mode-specific disclaimers
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

if user_mode == "Professional Mode":
    disclaimer = "⚠️ **Clinical Disclaimer:** This AI-based risk assessment tool is for clinical decision support only. Results should be interpreted within the full clinical context and do not replace comprehensive patient evaluation and clinical judgment."
elif user_mode == "Patient Mode":
    disclaimer = "⚠️ **Important:** This tool provides an estimate based on the information you entered. It's not a diagnosis and cannot replace your doctor's expertise. Always discuss your health concerns with your healthcare provider."
else:  # Family Mode
    disclaimer = "⚠️ **Family Note:** This tool is designed to help you understand potential heart disease risk. It's not a medical diagnosis. Your family member should always consult with their doctor about their health, especially if there are concerns."

st.markdown(f"""
<div style="text-align: center; color: #666; font-size: 14px; margin-top: 20px; background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px; padding: 15px;">
    <p>{disclaimer}</p>
</div>
""", unsafe_allow_html=True)