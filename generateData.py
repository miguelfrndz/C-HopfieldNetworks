import os
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

NORMALIZE = False

# If data folder does not exist, create it
if not os.path.exists('data'):
    os.makedirs('data')

def preprocess(img):
    thresh = img.mean()
    binary = img > thresh
    return binary.astype(int)

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
data = load_digits(n_class = 2)
X, y = data.data, data.target

# Preprocess the data
X = np.array([preprocess(x.reshape(8, 8)).flatten() for x in X])

# Split the dataset into training and testing sets (stratified)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

# Save the train and test sets to txt files
save_to_txt('data/train.txt', X_train, y_train)
save_to_txt('data/test.txt', X_test, y_test)

print("Data saved successfully!")