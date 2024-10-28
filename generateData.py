import os
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# If data folder does not exist, create it
if not os.path.exists('data'):
    os.makedirs('data')

# Function to save data to a txt file
def save_to_txt(filename, X, y):
    with open(filename, 'w') as f:
        # Write the number of instances and number of features
        f.write(f"{X.shape[0]}\n")
        f.write(f"{X.shape[1]}\n")
        # Write the data
        for i in range(X.shape[0]):
            features = ' '.join(map(str, X[i]))
            f.write(f"{features} {y[i]}\n")

# Load the breast cancer dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the training set
scaler = StandardScaler()
X_train_normalized = scaler.fit_transform(X_train)

# Normalize the testing set using the same scaler
X_test_normalized = scaler.transform(X_test)

# Save the train and test sets to txt files
save_to_txt('data/train.txt', X_train_normalized, y_train)
save_to_txt('data/test.txt', X_test_normalized, y_test)

print("Data saved successfully!")