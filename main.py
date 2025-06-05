# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv("heart.csv")

X = df.drop("target", axis=1)
y = df["target"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train a Decision Tree Classifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

# Predict and evaluate
y_pred_dt = dt.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print("\nClassification Report:\n", classification_report(y_test, y_pred_dt))

# Visualize the Decision Tree (limited depth for readability)
plt.figure(figsize=(20, 8))
plot_tree(dt, filled=True, feature_names=df.columns[:-1], max_depth=3, class_names=["No Disease", "Disease"])
plt.title("Decision Tree (Max Depth = 3 for visualization)")
plt.show()

# Analyze Overfitting (Train with max_depth)
train_accuracies = []
test_accuracies = []
depths = range(1, 11)

for d in depths:
    model = DecisionTreeClassifier(max_depth=d, random_state=42)
    model.fit(X_train, y_train)
    train_accuracies.append(model.score(X_train, y_train))
    test_accuracies.append(model.score(X_test, y_test))

plt.plot(depths, train_accuracies, label="Train Accuracy", marker="o")
plt.plot(depths, test_accuracies, label="Test Accuracy", marker="o")
plt.xlabel("Max Tree Depth")
plt.ylabel("Accuracy")
plt.title("Overfitting Analysis (Train vs Test Accuracy)")
plt.legend()
plt.grid(True)
plt.show()

# Train a Random Forest and compare accuracy
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("\nRandom Forest Classification Report:\n", classification_report(y_test, y_pred_rf))

# Feature Importances from Random Forest
importances = rf.feature_importances_
feature_names = df.columns[:-1]

# Visualize importances
plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=feature_names)
plt.title("Feature Importances (Random Forest)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.grid(True)
plt.show()

# Evaluate using Cross-Validation
rf_cv_scores = cross_val_score(rf, X_scaled, y, cv=5)
print("Random Forest Cross-Validation Accuracy Scores:", rf_cv_scores)
print("Mean CV Accuracy:", rf_cv_scores.mean())
