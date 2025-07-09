import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv("Task 3 and 4_Loan_Data.csv")

# Drop customer_id
df.drop("customer_id", axis=1, inplace=True)

# Features and target
X = df.drop("default", axis=1)
y = df["default"]

# Optional: Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model training (Logistic Regression)
model = LogisticRegression()
model.fit(X_train, y_train)

# Print evaluation
y_pred = model.predict(X_test)
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# 💡 Function to predict Expected Loss
def predict_expected_loss(borrower_info, exposure, recovery_rate=0.1):
    """
    borrower_info: dict with keys matching feature names (excluding 'default')
    exposure: loan exposure (e.g., loan amount)
    recovery_rate: assumed recovery (default is 10%)
    """
    input_df = pd.DataFrame([borrower_info])
    input_scaled = scaler.transform(input_df)
    prob_default = model.predict_proba(input_scaled)[0][1]
    expected_loss = prob_default * exposure * (1 - recovery_rate)
    return round(expected_loss, 2), round(prob_default * 100, 2)

# 🧪 Example usage
sample_borrower = {
    'credit_lines_outstanding': 2,
    'loan_amt_outstanding': 4000,
    'total_debt_outstanding': 7000,
    'income': 60000,
    'years_employed': 4,
    'fico_score': 610
}
exposure = 4000  # Loan amount
expected_loss, prob = predict_expected_loss(sample_borrower, exposure)

print(f"\n💰 Expected Loss: ${expected_loss}")
print(f"📉 Probability of Default: {prob}%")
