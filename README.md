# Predicting whether cell is Benign or Malignant using K-Nearest Neighbors

# Dataset Overview:
The WDBC dataset comprises 569 instances with 30 features each, including characteristics computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. The target variable, "Diagnosis," indicates whether the tumor is malignant (M) or benign (B).

# Data Preprocessing:

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
column_names = ["ID", "Diagnosis"] + [f"Feature_{i}" for i in range(1, 31)]
data = pd.read_csv(url, header=None, names=column_names)

# Drop unnecessary columns
data.drop("ID", axis=1, inplace=True)

# Convert Diagnosis to categorical variable
data["Diagnosis"] = data["Diagnosis"].astype("category").cat.codes

# Split into features (X) and target variable (y)
X = data.drop("Diagnosis", axis=1)
y = data["Diagnosis"]

# Standardize the data
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)
K-Nearest Neighbors Model:
python
Copy code
from sklearn.neighbors import KNeighborsClassifier

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_standardized, y, test_size=0.3, random_state=42)

# Build the KNN model
k_value = 5  # Example value, you may perform hyperparameter tuning
knn_model = KNeighborsClassifier(n_neighbors=k_value)
knn_model.fit(X_train, y_train)
Model Evaluation:
python
Copy code
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Make predictions on the testing set
y_pred = knn_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Confusion Matrix:\n{conf_matrix}")
Hyperparameter Tuning:
python
Copy code
from sklearn.model_selection import GridSearchCV

