import os
import joblib
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Credit Risk Prediction",
    page_icon="🏦",
    layout="wide"
)

# ---------------------------------------------------
# PATHS
# ---------------------------------------------------

BASE_DIR = os.path.dirname(__file__)

MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# ---------------------------------------------------
# THEME
# ---------------------------------------------------

st.markdown("""
<style>

.stApp{
    background:#0F172A;
    color:#F8FAFC;
}

.card{
    background:#1E293B;
    border-left:4px solid #B91C1C;
    padding:18px;
    border-radius:14px;
    margin-bottom:15px;
    color:#F8FAFC;
}

h1{
    color:#FFFFFF;
    font-weight:800;
    text-align:center;
}

h2{
    color:#FFFFFF;
}

h3{
    color:#F8FAFC;
}

.subtitle{
    color:#E2E8F0;
    text-align:center;
    font-size:18px;
}

.card{
    background:#1E293B;
    padding:18px;
    border-radius:14px;
    border-left:4px solid #B91C1C;
    margin-bottom:15px;
}

.stButton button{
    width:100%;
    background:#B91C1C;
    color:white;
    border:none;
    border-radius:10px;
    font-weight:600;
    height:45px;
}

.stButton button:hover{
    background:#991B1B;
}

/* Brighter labels */
label{
    color:#F8FAFC !important;
    font-weight:600;
}

/* Better metric cards */
div[data-testid="metric-container"]{
    background:#1E293B;
    border-radius:12px;
    padding:12px;
    border:1px solid #334155;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------

st.markdown("""
<div style="
background: linear-gradient(90deg, #1F2937, #111827);
padding:35px;
border-radius:18px;
border-left:6px solid #FF4B4B;
margin-bottom:30px;

display:flex;
flex-direction:column;
align-items:center;
justify-content:center;
">

<h1 style="
color:white;
font-size:42px;
font-weight:700;
margin:0;
text-align:center;
">
🏦 Credit Risk Prediction Dashboard
</h1>

<p style="
color:#D1D5DB;
font-size:18px;
margin:15px 0 20px 0;
text-align:center;
">
AI-Powered Loan Risk Assessment using Machine Learning
</p>

<div style="
width:75%;
height:1px;
background:#374151;
margin-bottom:20px;
"></div>

<div style="
display:flex;
justify-content:center;
align-items:center;
flex-wrap:wrap;
gap:12px;
">

<span style="
background:#374151;
padding:9px 18px;
border-radius:25px;
color:white;">
🤖 Random Forest
</span>

<span style="
background:#374151;
padding:9px 18px;
border-radius:25px;
color:white;">
📊 German Credit Dataset
</span>

<span style="
background:#374151;
padding:9px 18px;
border-radius:25px;
color:white;">
⚡ Real-Time Prediction
</span>

<span style="
background:#374151;
padding:9px 18px;
border-radius:25px;
color:white;">
🚀 Live Deployment
</span>

</div>

</div>
""", unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "🤖 Model",
        "Random Forest"
    )

with c2:
    st.metric(
        "📊 Dataset",
        "1000",
        "Records"
    )

with c3:
    st.metric(
        "🎯 Accuracy",
        "71.0%"
    )

with c4:
    st.metric(
        "📈 ROC-AUC",
        "65.6%"
    )


# ---------------------------------------------------
# APPLICANT INFORMATION
# ---------------------------------------------------

st.subheader("Applicant Information")

col1, col2 = st.columns(2)

with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )

    sex = st.selectbox(
        "Gender",
        ["male", "female"]
    )

    housing = st.selectbox(
        "Housing",
        ["own", "rent", "free"]
    )

    saving = st.selectbox(
        "Saving Account",
        ["little", "moderate", "quite rich", "rich"]
    )

    amount = st.number_input(
        "Credit Amount",
        min_value=0,
        value=3000,
        step=100
    )

with col2:

    job = st.selectbox(
        "Job Level",
        [0, 1, 2, 3]
    )

    checking = st.selectbox(
        "Checking Account",
        ["little", "moderate", "rich"]
    )

    duration = st.number_input(
        "Duration (Months)",
        min_value=1,
        value=12
    )

    purpose = st.selectbox(
        "Purpose",
        [
            "car",
            "radio/TV",
            "education",
            "furniture/equipment",
            "business",
            "domestic appliances",
            "repairs",
            "vacation/others"
        ]
    )

st.markdown("<br>", unsafe_allow_html=True)

predict = st.button("Predict Credit Risk")

# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------

if predict:

    input_df = pd.DataFrame({

        "Age":[age],
        "Sex":[sex],
        "Job":[job],
        "Housing":[housing],
        "Saving accounts":[saving],
        "Checking account":[checking],
        "Credit amount":[amount],
        "Duration":[duration],
        "Purpose":[purpose]

    })

    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0]

    bad_prob = probability[0] * 100
    good_prob = probability[1] * 100

    confidence = max(good_prob, bad_prob)

    st.markdown("---")

    # -------------------------
    # Result
    # -------------------------

    if prediction == 1:

        st.success("✅ GOOD CREDIT RISK")

        risk = "🟢 Low Risk"

    else:

        st.error("❌ BAD CREDIT RISK")

        risk = "🔴 High Risk"

    # -------------------------
    # Dashboard
    # -------------------------

    left, right = st.columns([1,1])

    with left:

        st.metric(
            "Confidence",
            f"{confidence:.1f}%"
        )

        st.metric(
            "Risk Level",
            risk
        )

        st.progress(confidence/100)

        fig = go.Figure(

            data=[

                go.Pie(

                    labels=[
                        "Good Risk",
                        "Bad Risk"
                    ],

                    values=[
                        good_prob,
                        bad_prob
                    ],

                    hole=0.65,

                    marker=dict(

                        colors=[
                            "#16A34A",
                            "#B91C1C"
                        ]

                    )

                )

            ]

        )

        fig.update_layout(

            template="plotly_dark",

            paper_bgcolor="#0F172A",

            font_color="white",

            margin=dict(
                l=20,
                r=20,
                t=20,
                b=20
            ),

            height=360,

            showlegend=True

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with right:

        st.markdown("### Applicant Summary")

        summary = pd.DataFrame({

            "Field":[

                "Age",
                "Gender",
                "Housing",
                "Job",
                "Amount",
                "Duration",
                "Purpose"

            ],

            "Value":[

                age,
                sex.title(),
                housing.title(),
                job,
                f"₹ {amount:,}",
                f"{duration} Months",
                purpose.title()

            ]

        })

        st.dataframe(

            summary,

            hide_index=True,

            use_container_width=True

        )

        st.info(

            f"""
Prediction Confidence : **{confidence:f}%**

Prediction Generated Successfully.
"""

        )

       # ---------------------------------------------------
# MODEL PERFORMANCE
# ---------------------------------------------------

st.markdown("---")

st.header("📊 Model Performance")

cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
roc_path = os.path.join(OUTPUT_DIR, "roc_curve.png")
feature_path = os.path.join(OUTPUT_DIR, "feature_importance.png")

# -------------------------------
# Top Row
# -------------------------------

col1, col2 = st.columns(2)

with col1:

    st.subheader("Confusion Matrix")

    if os.path.exists(cm_path):
        st.image(cm_path, width=420)
    else:
        st.warning("Image not found.")

with col2:

    st.subheader("ROC Curve")

    if os.path.exists(roc_path):
        st.image(roc_path, width=420)
    else:
        st.warning("Image not found.")

# -------------------------------
# Bottom Row
# -------------------------------

st.markdown("<br>", unsafe_allow_html=True)

st.subheader("Feature Importance")

left, center, right = st.columns([1,2,1])

with center:

    if os.path.exists(feature_path):
        st.image(feature_path, width=650)
    else:
        st.warning("Image not found.")

# ---------------------------------------------------
# ABOUT
# ---------------------------------------------------

# ---------------------------------------------------
# ABOUT
# ---------------------------------------------------

# ---------------------------------------------------
# ABOUT PROJECT
# ---------------------------------------------------

st.markdown("""
<div style="
background-color:#1F2937;
padding:30px;
border-radius:15px;
border-left:6px solid #FF4B4B;
margin-bottom:30px;
text-align:center;
">

<h2 style="color:white; margin-bottom:15px;">
🚀 About This Project
</h2>

<p style="
color:#EAEAEA;
font-size:16px;
line-height:1.8;
max-width:850px;
margin:auto;
">

This dashboard demonstrates an <b>end-to-end Machine Learning workflow</b>
for predicting <b>credit risk</b> using the <b>German Credit Dataset</b>.

The application analyzes an applicant's financial and demographic information
to determine whether they are likely to be a
<b style="color:#5BE37D;">Good Risk</b> or a
<b style="color:#FF6B6B;">Bad Risk</b> using a trained
<b>Random Forest Classifier</b>.

The dashboard also provides confidence scores, applicant summaries,
and model performance visualizations to support transparent and informed
credit risk assessment.

</p>

<hr style="
border:1px solid #374151;
margin:25px 0;
">

<div style="
display:flex;
justify-content:center;
align-items:center;
flex-wrap:wrap;
gap:12px;
">

<span style="
background:#374151;
padding:9px 18px;
border-radius:25px;
color:white;
font-size:14px;
font-weight:500;">
🧹 Data Cleaning
</span>

<span style="
background:#374151;
padding:9px 18px;
border-radius:25px;
color:white;
font-size:14px;
font-weight:500;">
⚙️ Feature Engineering
</span>

<span style="
background:#374151;
padding:9px 18px;
border-radius:25px;
color:white;
font-size:14px;
font-weight:500;">
🌳 Random Forest
</span>

<span style="
background:#374151;
padding:9px 18px;
border-radius:25px;
color:white;
font-size:14px;
font-weight:500;">
⚡ Real-Time Prediction
</span>

<span style="
background:#374151;
padding:9px 18px;
border-radius:25px;
color:white;
font-size:14px;
font-weight:500;">
📊 Interactive Dashboard
</span>

<span style="
background:#374151;
padding:9px 18px;
border-radius:25px;
color:white;
font-size:14px;
font-weight:500;">
🚀 Live Deployment
</span>

</div>

</div>
""", unsafe_allow_html=True)
# ---------------------------------------------------
# MODEL METRICS
# ---------------------------------------------------

m1, m2, m3, m4 = st.columns(4)

m1.metric("Accuracy", "71.0%")
m2.metric("Precision", "72.3%")
m3.metric("Recall", "95.0%")
m4.metric("ROC-AUC", "65.6%")

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.markdown("---")

st.markdown("""
<div style="text-align:center; padding:20px 0;">

<h4 style="margin-bottom:8px; color:white;">
🏦 Credit Risk Prediction Dashboard
</h4>

<p style="color:#D9D9D9; font-size:15px; margin-bottom:18px;">
Built with ❤️ using <b>Python</b>, <b>Scikit-learn</b> & <b>Streamlit</b>
</p>

<div style="margin-bottom:18px;">

<span style="background:#1F2937;
padding:8px 16px;
border-radius:25px;
margin:5px;
display:inline-block;
color:white;">
🐍 Python
</span>

<span style="background:#1F2937;
padding:8px 16px;
border-radius:25px;
margin:5px;
display:inline-block;
color:white;">
🤖 Machine Learning
</span>

<span style="background:#1F2937;
padding:8px 16px;
border-radius:25px;
margin:5px;
display:inline-block;
color:white;">
📊 Streamlit
</span>

</div>

<p style="margin-bottom:15px;">

<a href="https://github.com/gargiiiiiii14-png"
style="text-decoration:none; color:#4EA8FF; font-weight:bold;"
target="_blank">
GitHub
</a>

&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;

<a href="https://www.linkedin.com/in/gargi-rakshit-634047292/"
style="text-decoration:none; color:#4EA8FF; font-weight:bold;"
target="_blank">
LinkedIn
</a>

</p>

<p style="color:#8F8F8F; font-size:13px;">
© 2026 Gargi Rakshit
</p>

</div>
""", unsafe_allow_html=True)

