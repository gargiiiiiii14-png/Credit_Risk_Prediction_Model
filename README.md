# 🏦 Credit Risk Prediction using Machine Learning

## 📌 Project Overview

This project predicts whether a loan applicant is a **Good Credit Risk** or **Bad Credit Risk** using Machine Learning. The model is trained on the **German Credit Dataset** and deployed as an interactive **Streamlit web application**.

The application allows users to enter applicant details such as age, credit amount, loan duration, housing status, savings account, checking account, and loan purpose. Based on these inputs, the model predicts the applicant's credit risk along with prediction probabilities.

---

## 🎯 Objectives

* Perform data preprocessing and feature engineering.
* Train multiple Machine Learning classification models.
* Compare model performance using evaluation metrics.
* Improve the best model through hyperparameter tuning.
* Build an interactive web application using Streamlit.
* Deploy the application for real-time predictions.

---

## 📂 Dataset

**Dataset:** German Credit Dataset

The dataset contains information about loan applicants, including:

* Age
* Sex
* Job
* Housing
* Saving Accounts
* Checking Account
* Credit Amount
* Loan Duration
* Purpose of Loan

**Target Variable**

* **Good** → 1
* **Bad** → 0

---

## 🛠 Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Joblib
* Streamlit

---

## 🤖 Machine Learning Models

The following classification algorithms were implemented and compared:

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier

The Random Forest model was selected as the final model after comparing performance and applying hyperparameter tuning using **GridSearchCV**.

---

## 📊 Model Performance

| Model               |  Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| ------------------- | --------: | --------: | -----: | -------: | ------: |
| Random Forest       | **71.5%** |    75.15% | 88.57% |   81.31% |  65.84% |
| Logistic Regression |     64.5% |    80.00% | 65.71% |   72.16% |  66.07% |
| Decision Tree       |     63.0% |    73.24% | 74.29% |   73.76% |  55.48% |

---

## 📈 Model Evaluation

The project includes the following evaluation metrics:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC Score
* Classification Report
* Confusion Matrix
* ROC Curve
* Feature Importance Plot

---

## 🌐 Streamlit Web Application

The application provides an easy-to-use interface where users can:

* Enter applicant information
* Predict credit risk instantly
* View prediction probabilities
* View a probability bar chart
* Explore model evaluation visualizations

---

## 📁 Project Structure

```text
Credit_Risk_Prediction/
│
├── app.py
├── train_model.py
├── german_credit_data.csv
├── best_model.pkl
├── confusion_matrix.png
├── roc_curve.png
├── feature_importance.png
├── requirements.txt
├── README.md
```

---

## 🚀 How to Run Locally

### Clone the repository

```bash
git clone <your-github-repository-link>
```

### Navigate to the project directory

```bash
cd Credit_Risk_Prediction
```

### Install the required packages

```bash
pip install -r requirements.txt
```

### Train the model

```bash
python train_model.py
```

### Run the Streamlit application

```bash
streamlit run app.py
```

---

## 💡 Features

* Interactive user interface
* Real-time credit risk prediction
* Prediction probability visualization
* Machine Learning model comparison
* Hyperparameter tuning
* Feature importance analysis
* Easy deployment using Streamlit

---


## 👩‍💻 Author

**Gargi**

Machine Learning Project developed using Python, Scikit-learn, and Streamlit.

