import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Step 2: Load data
df = pd.read_csv("DSProject-CREDIT/CreditScoring-LoanDefault_Dataset.csv")
print(df.head())

# Preprocessing (simple version)
#check for missing values
print("Missing values before cleaning:")
print(df.isnull().sum())

# Fill missing values for Credit_History with "N/A"
df['Credit_History'] = df['Credit_History'].fillna("N/A")
print(df.isnull().sum())

#check for duplicate values
print("Duplicate values before cleaning:")
print(df.duplicated().sum())

#describe the dataset
print("Dataset description:")
print(df.describe())

# Save the output to a CSV file
df.to_csv("DSProject-CREDIT/Cleaned-Credit-Data.csv", index=False)
print("Cleaned data saved to 'Cleaned-Credit-Data.csv'")

#Train models (Decision Tree) 
# Encode categorical columns
categorical_cols = df.select_dtypes(include='object').columns
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))  # Convert to string to avoid errors
    print(f"Encoded {col} with classes: {le.classes_}")

# Features and target
target = "Loan_Default"
X = df.drop(columns=[target])
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Evaluate
y_pred = dt_model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, dt_model.predict_proba(X_test)[:, 1]))

# Visualize the Decision Tree using only matplotlib
plt.figure(figsize=(8,10))
from sklearn.tree import plot_tree
plot_tree(dt_model, filled=True, feature_names=X.columns, class_names=['No Default', 'Default'])
plt.title("Decision Tree Visualization")
plt.show()

