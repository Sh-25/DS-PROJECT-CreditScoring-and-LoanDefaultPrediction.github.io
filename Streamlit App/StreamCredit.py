# streamlit_credit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="Loan Default Predictor", layout="wide")
st.title("💳 Credit Scoring & Loan Default Prediction")
st.write("This app uses a Decision Tree model to predict loan defaults.")

# Cache loading and preprocessing
@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv("Cleaned-Credit-Data.csv")
    if "Customer_ID" in df.columns:
        df.drop(columns=["Customer_ID"], inplace=True)

    cat_cols = df.select_dtypes(include='object').columns
    num_cols = df.select_dtypes(include=np.number).columns

    df[cat_cols] = df[cat_cols].fillna("N/A")
    df[num_cols] = df[num_cols].fillna(df[num_cols].median(numeric_only=True))

    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    return df, cat_cols, label_encoders

df, cat_cols, label_encoders = load_and_prepare_data()
target = "Loan_Default"
X = df.drop(columns=[target])
y = df[target]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# --- PREDICTION SECTION ---
st.subheader("🔍 Predict a New Applicant")

user_input = {}
with st.form("prediction_form"):
    for col in X.columns:
        user_input[col] = st.number_input(
            f"{col}",
            float(X[col].min()),
            float(X[col].max()),
            float(X[col].mean()),
            step=1.0
        )
    submitted = st.form_submit_button("Predict")

if submitted:
    input_df = pd.DataFrame([user_input])
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    st.success(f"Prediction: {'🔴 Default' if pred else '🟢 No Default'}")
    st.info(f"Probability of Default: {prob:.2%}")

# --- SHAP EXPLAINABILITY ---
# st.subheader("🧠 SHAP Model Explainability")
# if st.checkbox("Show SHAP Summary Plot (First 100 test samples)"):
#     with st.spinner("Generating SHAP plots..."):
#         explainer = shap.Explainer(model, X_train)
#         shap_values = explainer(X_test[:100])
        
#         fig_summary, ax = plt.subplots(figsize=(10, 6))
#         shap.summary_plot(shap_values, X_test[:100], show=False)
#         st.pyplot(fig_summary)
