import pandas as pd
from scipy.stats import zscore
from sklearn.model_selection import train_test_split

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

# Apply z-score normalization to numerical columns
numerical_columns = ['foreheadwidthcm', 'foreheadheightcm', 'nosewide', 'noselong', 'lipsthin', 'distancenosetoliplong']
X_train_val[numerical_columns] = X_train_val[numerical_columns].apply(zscore)
X_test[numerical_columns] = X_test[numerical_columns].apply(zscore)

# Display the statistics after normalization
print("Statistics after z-score normalization:")
print(X_train_val[numerical_columns].describe())
print(X_test[numerical_columns].describe())
