# Predictive Modeling Using Machine Learning
# Algorithm: Random Forest Classifier

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Step 1: Load dataset
data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

print("PREDICTIVE MODELING USING MACHINE LEARNING")
print("-" * 45)

print("\nDataset loaded successfully!")
print("Number of rows:", X.shape[0])
print("Number of features:", X.shape[1])

print("\nFirst five rows:")
print(X.head())

# Step 2: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# Step 3: Create the Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Step 4: Train the model
model.fit(X_train, y_train)

print("\nModel trained successfully!")

# Step 5: Make predictions
y_pred = model.predict(X_test)

print("\nPredictions:")
print(y_pred[:10])

# Step 6: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy: {:.2f}%".format(accuracy * 100))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=data.target_names
    )
)

# Step 7: Display confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=data.target_names
)

disp.plot(cmap="Blues", values_format="d")
plt.title("Random Forest - Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# Step 8: Feature importance
importance = pd.Series(
    model.feature_importances_,
    index=data.feature_names
).sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 6))
importance.sort_values().plot(kind="barh")

plt.title("Top 10 Important Features")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

print("\nProject completed successfully!")
print("Graphs saved as image files.")
