from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Load sample dataset
X, y = load_iris(return_X_y=True)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")

print("Model saved successfully.")