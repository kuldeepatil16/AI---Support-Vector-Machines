import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.svm import SVC

# Load data from the files
separable_data = np.loadtxt('A2-ring-separable.txt')
merged_data = np.loadtxt('A2-ring-merged.txt')
test_data = np.loadtxt('A2-ring-test.txt')

# Separate features and class labels for separable and merged datasets
separable_features = separable_data[:, :2]
separable_labels = separable_data[:, 2]

merged_features = merged_data[:, :2]
merged_labels = merged_data[:, 2]

test_features = test_data[:, :2]
test_labels = test_data[:, 2]

# SVM-based scaling using MinMaxScaler
scaler = preprocessing.MinMaxScaler()
scaler.fit(separable_features)  # Fit scaler on training data

separable_features_scaled = scaler.transform(separable_features)
merged_features_scaled = scaler.transform(merged_features)
test_features_scaled = scaler.transform(test_features)

# Plotting the data for the separable dataset after SVM-based scaling
plt.figure(figsize=(8, 4))

plt.subplot(1, 3, 1)
plt.scatter(separable_features_scaled[:, 0], separable_features_scaled[:, 1], c=separable_labels, cmap='viridis', marker='o', s=10)
plt.title('Ring Separable Dataset (Scaled using SVM)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Plotting the data for the merged dataset after SVM-based scaling
plt.subplot(1, 3, 2)
plt.scatter(merged_features_scaled[:, 0], merged_features_scaled[:, 1], c=merged_labels, cmap='viridis', marker='o', s=10)
plt.title('Ring Merged Dataset (Scaled using SVM)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Plotting the data for the test dataset after SVM-based scaling
plt.subplot(1, 3, 3)
plt.scatter(test_features_scaled[:, 0], test_features_scaled[:, 1], c=test_labels, cmap='viridis', marker='o', s=10)
plt.title('Ring Test Dataset (Scaled using SVM)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()

# Training an SVM classifier on the scaled separable dataset
svm = SVC(kernel='linear')
svm.fit(separable_features_scaled, separable_labels)

# Make predictions on test data
predictions = svm.predict(test_features_scaled)

# Evaluate the model, compute accuracy, etc. 
# For example:
accuracy = np.mean(predictions == test_labels)
print(f"Accuracy: {accuracy * 100:.2f}%")
