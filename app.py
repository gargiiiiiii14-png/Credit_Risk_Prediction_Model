import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from PIL import Image

# -----------------------------
# Load trained model
# -----------------------------

model = joblib.load("best_model.pkl")

# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="Credit Risk Prediction",
    layout="wide"
)

# -----------------------------
# Sidebar
# -----------------------------

st.sidebar.title("🏦 Credit Risk Prediction")

st.sidebar.markdown("---")

st.sidebar.subheader("Project Information")

st.sidebar.write("**Dataset:** German Credit Data")

st.sidebar.write("**Machine Learning Models:**")
st.sidebar.write("- Logistic Regression")
st.sidebar.write("- Decision Tree")
st.sidebar.write("- Random Forest")

st.sidebar.markdown("---")

st.sidebar.subheader("Final Model")

st.sidebar.success("Random Forest Classifier")

st.sidebar.markdown("---")

st.sidebar.write(
    "Developed using Python, Scikit-learn and Streamlit."
)

st.title("🏦 Credit Risk Prediction Dashboard")
st.caption("Developed using Python, Scikit-learn and Streamlit")

st.write(
    """
    This application predicts whether a loan applicant is a **Good Risk**
    or **Bad Risk** using a Machine Learning model trained on the German
    Credit Dataset.
    """
)

st.markdown("---")

# -----------------------------
# User Inputs
# -----------------------------

col1, col2 = st.columns(2)

with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )

    sex = st.selectbox(
        "Sex",
        ["male", "female"]
    )

    job = st.selectbox(
        "Job",
        [0, 1, 2, 3]
    )

    housing = st.selectbox(
        "Housing",
        ["own", "rent", "free"]
    )

with col2:

    saving = st.selectbox(
        "Saving Account",
        ["little", "moderate", "quite rich", "rich"]
    )

    checking = st.selectbox(
        "Checking Account",
        ["little", "moderate", "rich"]
    )

    amount = st.number_input(
        "Credit Amount",
        value=3000
    )

    duration = st.number_input(
        "Duration (Months)",
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

st.markdown("---")

# -----------------------------
# Prediction
# -----------------------------
# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict Credit Risk"):

    input_data = pd.DataFrame({

        "Age": [age],
        "Sex": [sex],
        "Job": [job],
        "Housing": [housing],
        "Saving accounts": [saving],
        "Checking account": [checking],
        "Credit amount": [amount],
        "Duration": [duration],
        "Purpose": [purpose]

    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    good_probability = probability[1] * 100
    bad_probability = probability[0] * 100

    st.markdown("---")

    if prediction == 1:
        st.success("✅ Prediction: GOOD CREDIT RISK")
    else:
        st.error("❌ Prediction: BAD CREDIT RISK")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Good Risk Probability",
            f"{good_probability:.2f}%"
        )

    with col2:
        st.metric(
            "Bad Risk Probability",
            f"{bad_probability:.2f}%"
        )

    st.markdown("### 📊 Prediction Probability Chart")

    fig, ax = plt.subplots(figsize=(3.5, 2))

    categories = ["Good Risk", "Bad Risk"]
    values = [good_probability, bad_probability]

    bars = ax.bar(
    categories,
    values,
    color=["green", "red"]
)

    ax.set_ylim(0, 100)
    ax.set_ylabel("Probability (%)")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_title("Credit Risk Prediction Probability")

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 2,
            f"{height:.1f}%",
            ha="center"
        )

    st.pyplot(fig, use_container_width=False)

    st.markdown("---")

    if good_probability >= 80:
        st.success("🟢 Risk Level: Low Risk")
    elif good_probability >= 60:
        st.warning("🟡 Risk Level: Medium Risk")
    else:
        st.error("🔴 Risk Level: High Risk")

        st.markdown("---")
st.subheader("📋 Applicant Summary")

prediction_text = "GOOD CREDIT RISK" if prediction == 1 else "BAD CREDIT RISK"

summary = pd.DataFrame({
    "Field": [
        "Age",
        "Sex",
        "Housing",
        "Credit Amount",
        "Duration (Months)",
        "Purpose",
        "Prediction",
        "Confidence"
    ],
    "Value": [
        age,
        sex.title(),
        housing.title(),
        amount,
        duration,
        purpose.title(),
        prediction_text,
        f"{max(good_probability, bad_probability):.2f}%"
    ]
})

st.table(summary)

   

st.header("📊 Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.image(
        "outputs/confusion_matrix.png",
        caption="Confusion Matrix",
        use_container_width=True
    )

with col2:
    st.image(
        "outputs/roc_curve.png",
        caption="ROC Curve",
        use_container_width=True
    )

st.image(
    "outputs/feature_importance.png",
    caption="Feature Importance",
    use_container_width=True
)
