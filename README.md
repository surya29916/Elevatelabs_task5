# ElevateLabs_Task5

## 📊 Dataset Overview

The dataset contains medical records of patients, including features like age, cholesterol, chest pain type, resting blood pressure, etc. The target variable is:
- `1`: Patient has heart disease
- `0`: No heart disease

---

## ✅ Steps Performed

### 1. Import and Preprocess the Dataset
- Loaded the dataset using `pandas`.
- Split the data into features (`X`) and target (`y`).
- Standardized numerical features using `StandardScaler`.

### 2. Train a Decision Tree Classifier
- Used `DecisionTreeClassifier` from `sklearn.tree` to fit the model.
- Evaluated using accuracy score and classification report.
- Visualized the decision tree (limited to `max_depth=3` for clarity).

### 3. Analyze Overfitting and Control Tree Depth
- Trained decision trees with depth from 1 to 10.
- Plotted train vs. test accuracy to identify overfitting patterns.
- Helped in selecting the best depth that balances bias and variance.

### 4. Train a Random Forest and Compare Accuracy
- Used `RandomForestClassifier` from `sklearn.ensemble`.
- Compared its performance against the Decision Tree.
- Observed improved performance due to ensemble averaging.

### 5. Interpret Feature Importances
- Extracted feature importances from the Random Forest model.
- Visualized them using a bar plot to understand which features most influence predictions.

### 6. Evaluate using Cross-Validation
- Used 5-fold cross-validation on the Random Forest model.
- Reported average accuracy for a more reliable evaluation.

---

## 🧠 Interpretation

- **Random Forest** generally performs better than a single Decision Tree due to reduced variance.
- Features like `cp` (chest pain type), `thalach` (max heart rate), and `oldpeak` were among the most important.
- Overfitting can be controlled in decision trees by limiting the `max_depth`.

---

## 🛠️ Tools & Libraries Used

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn

---
