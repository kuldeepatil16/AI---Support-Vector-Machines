import numpy as np
import matplotlib.pyplot as plt

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

# Z-score Normalization (Standardization)
separable_features = (separable_features - np.mean(separable_features, axis=0)) / np.std(separable_features, axis=0)
merged_features = (merged_features - np.mean(merged_features, axis=0)) / np.std(merged_features, axis=0)
test_features = (test_features - np.mean(test_features, axis=0)) / np.std(test_features, axis=0)

# Plotting the data for the separable dataset after normalization
plt.figure(figsize=(8, 4))

plt.subplot(1, 3, 1)
plt.scatter(separable_features[:, 0], separable_features[:, 1], c=separable_labels, cmap='viridis', marker='o', s=10)
plt.title('Ring Separable Dataset (Normalized)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Plotting the data for the merged dataset after normalization
plt.subplot(1, 3, 2)
plt.scatter(merged_features[:, 0], merged_features[:, 1], c=merged_labels, cmap='viridis', marker='o', s=10)
plt.title('Ring Merged Dataset (Normalized)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Plotting the data for the test dataset after normalization
plt.subplot(1, 3, 3)
plt.scatter(test_features[:, 0], test_features[:, 1], c=test_labels, cmap='viridis', marker='o', s=10)
plt.title('Ring Test Dataset (Normalized)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()

# Statistical analysis for separable dataset after normalization
mean_separable_normalized = np.mean(separable_features, axis=0)
variance_separable_normalized = np.var(separable_features, axis=0)

# Statistical analysis for merged dataset after normalization
mean_merged_normalized = np.mean(merged_features, axis=0)
variance_merged_normalized = np.var(merged_features, axis=0)

# Statistical analysis for test dataset after normalization
mean_test_normalized = np.mean(test_features, axis=0)
variance_test_normalized = np.var(test_features, axis=0)

print("Separable Dataset after normalization:")
print("Mean:", mean_separable_normalized)
print("Variance:", variance_separable_normalized)

print("Merged Dataset after normalization:")
print("Mean:", mean_merged_normalized)
print("Variance:", variance_merged_normalized)

print("Test Dataset after normalization:")
print("Mean:", mean_test_normalized)
print("Variance:", variance_test_normalized)