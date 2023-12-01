import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load the gender dataset with the correct delimiter ('\t')
data_gender = pd.read_csv('A2-gender.txt', delimiter='\t', names=['longhair', 'foreheadwidthcm', 'foreheadheightcm', 'nosewide', 'noselong', 'lipsthin', 'distancenosetoliplong', 'gender'])

# Display basic information about the dataset
print(data_gender.info())

# Remove lines with unknown values
data_gender.replace('unknown', pd.NA, inplace=True)
data_gender.dropna(inplace=True)

# Map categorical values to numeric representations
data_gender['gender'] = data_gender['gender'].map({'Male': 1, 'Female': 0})

# Define input features (X) and target variable (y)
X_gender = data_gender.drop(columns=['gender'])  # Input features
y_gender = data_gender['gender']  # Target variable

# Split the data into training/validation sets (80%) and test set (20%) with shuffling
X_train_val, X_test, y_train_val, y_test = train_test_split(X_gender, y_gender, test_size=0.2, shuffle=True, random_state=42)

# Initialize the SVM classifier
svm_classifier = SVC(kernel='linear', random_state=42)  # Hany we can choose different kernels (e.g., 'linear', 'rbf', 'poly', etc.)

# Train the SVM classifier
svm_classifier.fit(X_train_val, y_train_val)

# Predict on the test set
y_pred = svm_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of SVM Classifier: {accuracy:.2f}")

# Display classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))
