
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 1. Load the dataset
data = pd.read_csv("weather.csv")

# 2. Check required columns
features = [
    "temperature",
    "humidity",
    "wind_speed",
    "pressure"
]

target = "weather"

if data[features + [target]].isnull().any().any():
    raise ValueError("Dataset contains missing values. Please clean weather.csv.")

# 3. Separate input features and target
X = data[features]
y = data[target]

# 4. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# 5. Create the machine learning pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    ))
])

# 6. Train the model
model.fit(X_train, y_train)

# 7. Evaluate the model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\n===== AI WEATHER MODEL RESULTS =====")
print(f"Test accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))

# 8. Save the trained model
joblib.dump(model, "weather_model.pkl")

print("\nModel saved successfully as weather_model.pkl")
print("Features:", features)
print("Classes:", model.classes_)
