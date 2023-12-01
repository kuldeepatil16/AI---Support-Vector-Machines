import numpy as np
import matplotlib.pyplot as plt

# Load data from the files
separable_data = np.loadtxt('A2-ring-separable.txt')
merged_data = np.loadtxt('A2-ring-merged.txt')

# Separate features and class labels for separable and merged datasets
separable_features = separable_data[:, :2]
separable_labels = separable_data[:, 2]

merged_features = merged_data[:, :2]
merged_labels = merged_data[:, 2]

# Plotting the data for the separable dataset
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.scatter(separable_features[:, 0], separable_features[:, 1], c=separable_labels, cmap='viridis', marker='o', s=10)
plt.title('Ring Separable Dataset')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Plotting the data for the merged dataset
plt.subplot(1, 2, 2)
plt.scatter(merged_features[:, 0], merged_features[:, 1], c=merged_labels, cmap='viridis', marker='o', s=10)
plt.title('Ring Merged Dataset')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()

# Statistical analysis for separable dataset
mean_separable = np.mean(separable_features, axis=0)
variance_separable = np.var(separable_features, axis=0)

# Statistical analysis for merged dataset
mean_merged = np.mean(merged_features, axis=0)
variance_merged = np.var(merged_features, axis=0)

print("Separable Dataset:")
print("Mean:", mean_separable)
print("Variance:", variance_separable)

print("\nMerged Dataset:")
print("Mean:", mean_merged)
print("Variance:", variance_merged)