import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

# Z-score Normalization (Standardization)
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Split the normalized data into training and test sets (80% - 20%)
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, shuffle=True, random_state=42)

# Display the shapes of the resulting sets
print("Training set shape - X_train:", X_train.shape, "y_train:", y_train.shape)
print("Test set shape - X_test:", X_test.shape, "y_test:", y_test.shape)