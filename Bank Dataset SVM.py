import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data = pd.read_csv('bank-additional-full.csv', delimiter=';')

# Display basic information about the dataset
print(data.info())

# Selecting the first 20 features and the target column
selected_features = data.iloc[:, :20]  # The first 20 columns are the features
target_column = data['y']  # Target column ('y': subscribed or not)

# Handle missing values (replace 'unknown' with NaN)
data.replace('unknown', pd.NA, inplace=True)

# Identify categorical columns for encoding
categorical_columns = data.select_dtypes(include=['object']).columns.tolist()

# One-hot encode categorical columns
data_encoded = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# Define input features and prediction feature
X = data_encoded.drop(columns=['y_yes'])  # Input features
y = data_encoded['y_yes']  # Prediction feature ('y_yes': subscribed or not)

# Split the data into training and test sets (80% - 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

# SVM Classifier
svm_classifier = SVC(kernel='rbf', random_state=42)  # You can choose different kernels and parameters

# Train the SVM classifier
svm_classifier.fit(X_train, y_train)

# Predict using the trained classifier
y_pred = svm_classifier.predict(X_test)

# Assessing the accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))
