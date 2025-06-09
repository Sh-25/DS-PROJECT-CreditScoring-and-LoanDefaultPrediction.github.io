# 📊 Credit Scoring and Loan Default Prediction

## 🔍 Project Overview
This project aims to predict the likelihood of loan default using historical customer data and machine learning.
By combining predictive modeling with a user-friendly web app and interactive dashboard, 
it helps financial institutions make smarter,faster, and more accurate lending decisions.

The project includes:
- A trained **Decision Tree classifier**
- A **Streamlit web app** for real-time predictions
- A **Power BI dashboard** for business analysis

- ## 📁 Dataset
**File**: `CreditScoring-LoanDefault_Dataset.csv`  
Contains customer information including:
- Age, Gender
- Income, Loan Amount, Loan Tenure
- Credit History, Marital Status, Employment Type  
**Target Variable**: `Loan_Default` (1 = Default, 0 = No Default)

## 🔧 Data Preprocessing
- Missing values handled with median (numeric) and 'N/A' (categorical)
- Categorical columns label-encoded
- Train-test split: 80% training, 20% testing
-  Saved cleaned dataset to `Cleaned-Credit-Data.csv`

-  ## 🤖 Model Used
**Decision Tree Classifier**
- Simple and interpretable
- Performs well on structured data
- Evaluated using:
  - Accuracy
  - Confusion Matrix
  - ROC AUC Score (~0.86)
 
## 🌲 Decision Tree Visualization
The decision tree was visualized to show how the model predicts loan default using feature-based decision paths.
This enhances transparency and model explainability for stakeholders.

## 🖥️ Streamlit App
Implemented in `StreamCredit.py`  
Features:
- Real-time prediction form with sliders for each feature
- Displays:
  - Binary result (🔴 Default / 🟢 No Default)
  - Probability of default
- Caches data loading and encodes inputs using LabelEncoders

Run with:
streamlit run StreamCredit.py

## 📊 Power BI Dashboard
The Power BI dashboard enables:
- Real-time visualization of default patterns
- Filtering by gender, credit history, income, etc.
- Bar, pie, and line charts for intuitive insights
- Business-friendly risk exploration

## 🧾 Conclusion
This project demonstrates the practical application of data science in the financial domain by building an end-to-end credit scoring and loan default prediction system. Using a Decision Tree classifier, the model identifies risk factors such as credit history and income level to accurately predict defaults.

The integration of a Streamlit web app enables real-time risk evaluation for new applicants, while the Power BI dashboard offers valuable business insights through dynamic visualizations. Together, these tools support smarter, faster, and more transparent lending decisions.

## 🛠️ Tools & Libraries Used

Programming Language :
Python – Core language for model development and app building

Data Processing:
pandas – Data manipulation and cleaning

numpy – Numerical operations

re – Text preprocessing with regular expressions

Visualization:
matplotlib – Static visualizations (decision tree, trends)
plotly – Interactive visualizations in the Streamlit app
Power BI – Business dashboard for exploratory insights

Machine Learning :
scikit-learn – Model building and evaluation (Decision Tree, metrics)
LabelEncoder – Encoding categorical variables for ML

Model Evaluation
confusion_matrix, classification_report, roc_auc_score – Used to evaluate model performance

Web App Development:
Streamlit – For creating the interactive real-time prediction app
st.cache_data – For efficient loading and processing of CSV data in Streamlit

Other
langdetect – (if used in future sentiment features or multilingual credit scoring)

❤️  Author: [Shruti Badagandi]
