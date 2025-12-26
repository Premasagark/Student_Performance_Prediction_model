import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score



# dataset
df = pd.read_csv("dataset/student_data(1).csv")

# data preprocessing for input and output
X = df[["study_hours", "attendance", "internal_marks", "practice_hours"]]
y = df["performance"]

    # Encode labels 
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42
)

# Logistic Regression ML Algorithm 
model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000
)
model.fit(X_train, y_train)


# displaying Accuracy Score
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")


# save model in model directory

with open("model/model.pkl", "wb") as f:
    pickle.dump(
        {
            "model": model,
            "scaler": scaler,
            "label_encoder": label_encoder
        },
        f
    )

print("Model saved successfully in model directory")
