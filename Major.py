import numpy as np
from scipy.linalg import pinv
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE
import time

# Load dataset
dataset = pd.read_csv("/content/Scenario-B-merged_5s.csv")
print(dataset.shape)
# Feature and label extraction
targets = ['label']
feature_names = [col for col in dataset.columns if col not in targets]

# Handle categorical features
def cat_conv(data):
    data['Source IP'] = data['Source IP'].apply(hash).astype('float64')
    data[' Destination IP'] = data[' Destination IP'].apply(hash).astype('float64')
    return data

clean_data = cat_conv(dataset)

# Splitting data into X and y
def X_y_creation(dataset):
    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]
    return X, y

X, y_multi = X_y_creation(clean_data)

# Handling missing and infinite values
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.dropna(axis=0, inplace=True)
y_multi = y_multi.loc[X.index].reset_index(drop=True)

# Label Encoding
label_encoder = LabelEncoder()
y_multi_encoded = label_encoder.fit_transform(y_multi)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_multi_encoded, test_size=0.2, random_state=42, stratify=y_multi_encoded
)
print(X_train.shape)
print(y_train.shape)
# Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Feature selection
clf = ExtraTreesClassifier(n_estimators=50)
clf.fit(X_train, y_train)
model = SelectFromModel(clf, prefit=True)
X_train = model.transform(X_train)
X_test = model.transform(X_test)

print(X_train.shape)
print(y_train.shape)
# Ridge Regression-Based ELM
class ELM_Ridge:
    def __init__(self, hidden_units, activation_function, x, y, lambda_reg, elm_type, random_type='normal'):
        self.hidden_units = hidden_units
        self.activation_function = activation_function
        self.random_type = random_type
        self.x = x
        self.y = y
        self.class_num = len(np.unique(self.y))
        self.lambda_reg = lambda_reg
        self.elm_type = elm_type

        if self.elm_type == 'clf':
            self.encoder = LabelEncoder()
            self.y_encoded = self.encoder.fit_transform(self.y)
            self.y_temp = np.eye(self.class_num)[self.y_encoded]  # One-hot encoding

        if self.random_type == 'uniform':
            self.W = np.random.uniform(low=-1, high=1, size=(self.hidden_units, self.x.shape[1]))
            self.b = np.random.uniform(low=-1, high=1, size=(self.hidden_units, 1))
        else:  # normal distribution
            self.W = np.random.normal(loc=0, scale=0.5, size=(self.hidden_units, self.x.shape[1]))
            self.b = np.random.normal(loc=0, scale=0.5, size=(self.hidden_units, 1))

    def __input2hidden(self, x):
        H = np.dot(self.W, x.T) + self.b
        if self.activation_function == 'sigmoid':
            return 1 / (1 + np.exp(-H)), H
        elif self.activation_function == 'relu':
            return np.maximum(0, H), H
        elif self.activation_function == 'sin':
            return np.sin(H), H
        elif self.activation_function == 'tanh':
            return np.tanh(H), H
        elif self.activation_function == 'leaky_relu':
            return np.maximum(0, H) + 0.1 * np.minimum(0, H), H
        else:
            raise ValueError("Unsupported activation function")

    def __hidden2output(self, H):
        return np.dot(H.T, self.beta)

    def softmax(self, output):
        exp_values = np.exp(output - np.max(output, axis=1, keepdims=True))  # Stability trick (subtract max)
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)  # Normalize to get probabilities
        return probabilities

    def fit(self):
        start_time = time.time()
        H, _ = self.__input2hidden(self.x)

        # Ridge regression weight calculation
        H_t_H = np.dot(H, H.T) + self.lambda_reg * np.eye(H.shape[0])

        self.beta = np.dot(pinv(H_t_H), np.dot(H, self.y_temp))  # Using pseudo-inverse

        end_time = time.time()
        self.train_time = end_time - start_time

        train_output = self.__hidden2output(H)

        if self.elm_type == 'clf':
            predicted_labels = np.argmax(train_output, axis=1)
            self.train_score = np.mean(predicted_labels == self.y_encoded)

        return self.beta, self.train_score, self.train_time

    def predict(self, x):
        H, hidden_activations = self.__input2hidden(x)
        output = self.__hidden2output(H)

        # Calculate probabilities (assuming softmax activation for classification)
        probabilities = self.softmax(output)

        if self.elm_type == 'clf':
            return probabilities, H, hidden_activations, output
        return output

    def score(self, x, y):
        y_pred = self.predict(x)[0]  # Get probabilities for scoring
        y_pred_labels = np.argmax(y_pred, axis=1)  # Convert probabilities to class labels
        return np.mean(y_pred_labels == y)


# Main Execution
if __name__ == "__main__":
    elm_ridge_model = ELM_Ridge(
        hidden_units=100, activation_function='sigmoid',
        x=X_train, y=y_train, lambda_reg=0.1, elm_type='clf'
    )

    beta, train_score, train_time = elm_ridge_model.fit()
    print(f"Training Accuracy: {train_score * 100:.2f}%")
    print(f"Training Time: {train_time:.4f} seconds")

    test_score = elm_ridge_model.score(X_test, y_test)
    print(f"Test Accuracy: {test_score * 100:.2f}%")

    # Example: Print prediction details for a single sample
    sample_index = 17  # You can change this index to test different samples
    sample_input = X_test[sample_index].reshape(1, -1)  # Ensure it's a 2D array
    sample_actual = y_test[sample_index]  # Get the actual class label

    # Get the model's prediction and probabilities for all classes
    print("\n--- Starting Prediction Calculation ---")

    print("\nStep 1: Sample Input (Features)")
    print(f"Sample Input: {sample_input}")

    probabilities, H, hidden_activations, output = elm_ridge_model.predict(sample_input)

    print("\nStep 2: Hidden Layer Activations (H)")
    print(f"Hidden Layer Activations (H): {hidden_activations}")

    print("\nStep 3: Weights (W) Between Input and Hidden Layer")
    print(f"Weights (W): {elm_ridge_model.W}")

    print("\nStep 4: Biases (b) for Hidden Layer")
    print(f"Biases (b): {elm_ridge_model.b}")

    print("\nStep 5: Ridge Regression Coefficients (Beta)")
    print(f"Beta (Ridge Regression Coefficients): {elm_ridge_model.beta}")

    print("\nStep 6: Output Before Softmax (Raw Output)")
    print(f"Output Before Softmax: {output}")

    print("\nStep 7: Calculated Probabilities (Softmax Output)")
    print(f"Probabilities: {probabilities}")

    # Convert prediction if using one-hot encoding
    predicted_label = np.argmax(probabilities)  # The label with the highest probability
    predicted_class_name = label_encoder.inverse_transform([predicted_label])[0]
    actual_class_name = label_encoder.inverse_transform([sample_actual])[0]


    print("\nClass Probabilities for the Sample:")
    for i, prob in enumerate(probabilities[0]):
        class_name = label_encoder.inverse_transform([i])[0]  # Get class name for each index
        print(f"Class: {class_name}, Probability: {prob:.4f}")
    # Print final prediction and actual values
    print(f"\nPredicted Label: {predicted_label} ({predicted_class_name})")
    print(f"Actual Label: {sample_actual} ({actual_class_name})")

    # Get feature mask (True for selected features, False for removed ones)
selected_features_mask = model.get_support()

# Get original feature names
original_features = X.columns  # Assuming X is the original feature set before selection

# Get the names of selected features
selected_features = original_features[selected_features_mask]

# Print selected features
print("Selected Features After Feature Selection:")
print(selected_features.tolist())

# Get the selected feature mask (True for selected features, False for unselected)
selected_features_mask = model.get_support()

# Get the original feature names
original_features = X.columns  # Assuming X is the original DataFrame before feature selection

# Get the names of selected features
selected_features = original_features[selected_features_mask]

# Print the selected features
print("Selected Features After Feature Selection:")
print(selected_features.tolist())

from sklearn.metrics import confusion_matrix, classification_report


# After calculating test accuracy, let's get the predictions for the test set
y_pred = elm_ridge_model.predict(X_test)[0]  # Get probabilities
y_pred_labels = np.argmax(y_pred, axis=1)  # Convert probabilities to predicted labels

# Get the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_labels)

# Print the confusion matrix
print("\nConfusion Matrix:")
print(conf_matrix)

# Optionally, you can also print a more detailed classification report
# This gives precision, recall, f1-score for each class
print("\nClassification Report:")
print(classification_report(y_test, y_pred_labels, target_names=label_encoder.classes_))

# You can visualize the confusion matrix using a heatmap (optional but helpful)
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

import numpy as np
from scipy.linalg import pinv
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE
import time

# Load dataset
dataset = pd.read_csv("/content/Scenario-B-merged_5s.csv")

# Feature and label extraction
targets = ['Label', 'Label.1']
feature_names = [col for col in dataset.columns if col not in targets]

# Handle categorical features
def cat_conv(data):
    data['Source IP'] = data['Source IP'].apply(hash).astype('float64')
    data[' Destination IP'] = data[' Destination IP'].apply(hash).astype('float64')
    return data

clean_data = cat_conv(dataset)

# Splitting data into X and y
def X_y_creation(dataset):
    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]
    return X, y

X, y_multi = X_y_creation(clean_data)

# Handling missing and infinite values
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.dropna(axis=0, inplace=True)
y_multi = y_multi.loc[X.index].reset_index(drop=True)

# Label Encoding
label_encoder = LabelEncoder()
y_multi_encoded = label_encoder.fit_transform(y_multi)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_multi_encoded, test_size=0.2, random_state=42, stratify=y_multi_encoded
)

# Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Feature selection
clf = ExtraTreesClassifier(n_estimators=50)
clf.fit(X_train, y_train)
model = SelectFromModel(clf, prefit=True)
X_train = model.transform(X_train)
X_test = model.transform(X_test)


# Ridge Regression-Based ELM
class ELM_Ridge:
    def __init__(self, hidden_units, activation_function, x, y, lambda_reg, elm_type, random_type='normal'):
        self.hidden_units = hidden_units
        self.activation_function = activation_function
        self.random_type = random_type
        self.x = x
        self.y = y
        self.class_num = len(np.unique(self.y))
        self.lambda_reg = lambda_reg
        self.elm_type = elm_type

        if self.elm_type == 'clf':
            self.encoder = LabelEncoder()
            self.y_encoded = self.encoder.fit_transform(self.y)
            self.y_temp = np.eye(self.class_num)[self.y_encoded]  # One-hot encoding

        if self.random_type == 'uniform':
            self.W = np.random.uniform(low=-1, high=1, size=(self.hidden_units, self.x.shape[1]))
            self.b = np.random.uniform(low=-1, high=1, size=(self.hidden_units, 1))
        else:  # normal distribution
            self.W = np.random.normal(loc=0, scale=0.5, size=(self.hidden_units, self.x.shape[1]))
            self.b = np.random.normal(loc=0, scale=0.5, size=(self.hidden_units, 1))

    def __input2hidden(self, x):
        H = np.dot(self.W, x.T) + self.b

        if self.activation_function == 'sigmoid':
            return 1 / (1 + np.exp(-H))
        elif self.activation_function == 'relu':
            return np.maximum(0, H)
        elif self.activation_function == 'sin':
            return np.sin(H)
        elif self.activation_function == 'tanh':
            return np.tanh(H)
        elif self.activation_function == 'leaky_relu':
            return np.maximum(0, H) + 0.1 * np.minimum(0, H)
        else:
            raise ValueError("Unsupported activation function")

    def fit(self):
        start_time = time.time()
        H = self.__input2hidden(self.x)

        # Ridge regression weight calculation
        H_t_H = np.dot(H, H.T) + self.lambda_reg * np.eye(H.shape[0])
        self.beta = np.dot(pinv(H_t_H), np.dot(H, self.y_temp))  # Using pseudo-inverse

        end_time = time.time()
        self.train_time = end_time - start_time

        train_output = np.dot(H.T, self.beta)

        if self.elm_type == 'clf':
            predicted_labels = np.argmax(train_output, axis=1)
            self.train_score = np.mean(predicted_labels == self.y_encoded)

        return self.beta, self.train_score, self.train_time

    def predict(self, x):
        H = self.__input2hidden(x)
        output = np.dot(H.T, self.beta)

        if self.elm_type == 'clf':
            return np.argmax(output, axis=1)
        return output

    def score(self, x, y):
        y_pred = self.predict(x)
        return np.mean(y_pred == y)


# Grid Search for optimal lambda_reg
lambda_values = [0.01, 0.1, 1, 10, 100]
best_lambda = None
best_score = 0

for lambda_reg in lambda_values:
    elm_ridge_model = ELM_Ridge(
        hidden_units=100, activation_function='sigmoid',
        x=X_train, y=y_train, lambda_reg=lambda_reg, elm_type='clf'
    )

    # Fit the model and get the training score
    beta, train_score, _ = elm_ridge_model.fit()

    if train_score > best_score:
        best_score = train_score
        best_lambda = lambda_reg

print(f"Best lambda_reg: {best_lambda} with training accuracy: {best_score * 100:.2f}%")

# Training the final model with the best lambda_reg
final_elm_ridge_model = ELM_Ridge(
    hidden_units=100, activation_function='sigmoid',
    x=X_train, y=y_train, lambda_reg=best_lambda, elm_type='clf'
)

beta, train_score, train_time = final_elm_ridge_model.fit()
print(f"Training Accuracy: {train_score * 100:.2f}%")
print(f"Training Time: {train_time:.4f} seconds")

# Test the model on the test set
test_score = final_elm_ridge_model.score(X_test, y_test)
print(f"Test Accuracy: {test_score * 100:.2f}%")

import numpy as np
from scipy.linalg import pinv
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE
import time

# Load dataset
dataset = pd.read_csv("/content/Scenario-B-merged_5s.csv")

# Feature and label extraction
targets = ['Label', 'Label.1']
feature_names = [col for col in dataset.columns if col not in targets]

# Handle categorical features
def cat_conv(data):
    data['Source IP'] = data['Source IP'].apply(hash).astype('float64')
    data[' Destination IP'] = data[' Destination IP'].apply(hash).astype('float64')
    return data

clean_data = cat_conv(dataset)

# Splitting data into X and y
def X_y_creation(dataset):
    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]
    return X, y

X, y_multi = X_y_creation(clean_data)

# Handling missing and infinite values
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.dropna(axis=0, inplace=True)
y_multi = y_multi.loc[X.index].reset_index(drop=True)

# Label Encoding
label_encoder = LabelEncoder()
y_multi_encoded = label_encoder.fit_transform(y_multi)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_multi_encoded, test_size=0.2, random_state=42, stratify=y_multi_encoded
)

# Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Feature selection
clf = ExtraTreesClassifier(n_estimators=50)
clf.fit(X_train, y_train)
model = SelectFromModel(clf, prefit=True)
X_train = model.transform(X_train)
X_test = model.transform(X_test)


# Ridge Regression-Based ELM
class ELM_Ridge:
    def __init__(self, hidden_units, activation_function, x, y, lambda_reg, elm_type, random_type='normal'):
        self.hidden_units = hidden_units
        self.activation_function = activation_function
        self.random_type = random_type
        self.x = x
        self.y = y
        self.class_num = len(np.unique(self.y))
        self.lambda_reg = lambda_reg
        self.elm_type = elm_type

        if self.elm_type == 'clf':
            self.encoder = LabelEncoder()
            self.y_encoded = self.encoder.fit_transform(self.y)
            self.y_temp = np.eye(self.class_num)[self.y_encoded]  # One-hot encoding

        if self.random_type == 'uniform':
            self.W = np.random.uniform(low=-1, high=1, size=(self.hidden_units, self.x.shape[1]))
            self.b = np.random.uniform(low=-1, high=1, size=(self.hidden_units, 1))
        else:  # normal distribution
            self.W = np.random.normal(loc=0, scale=0.5, size=(self.hidden_units, self.x.shape[1]))
            self.b = np.random.normal(loc=0, scale=0.5, size=(self.hidden_units, 1))

    def __input2hidden(self, x):
        H = np.dot(self.W, x.T) + self.b

        if self.activation_function == 'sigmoid':
            return 1 / (1 + np.exp(-H))
        elif self.activation_function == 'relu':
            return np.maximum(0, H)
        elif self.activation_function == 'sin':
            return np.sin(H)
        elif self.activation_function == 'tanh':
            return np.tanh(H)
        elif self.activation_function == 'leaky_relu':
            return np.maximum(0, H) + 0.1 * np.minimum(0, H)
        else:
            raise ValueError("Unsupported activation function")

    def fit(self):
        start_time = time.time()
        H = self.__input2hidden(self.x)

        # Ridge regression weight calculation
        H_t_H = np.dot(H, H.T) + self.lambda_reg * np.eye(H.shape[0])
        self.beta = np.dot(pinv(H_t_H), np.dot(H, self.y_temp))  # Using pseudo-inverse

        end_time = time.time()
        self.train_time = end_time - start_time

        train_output = np.dot(H.T, self.beta)

        if self.elm_type == 'clf':
            predicted_labels = np.argmax(train_output, axis=1)
            self.train_score = np.mean(predicted_labels == self.y_encoded)

        return self.beta, self.train_score, self.train_time

    def predict(self, x):
        H = self.__input2hidden(x)
        output = np.dot(H.T, self.beta)

        if self.elm_type == 'clf':
            return np.argmax(output, axis=1)
        return output

    def score(self, x, y):
        y_pred = self.predict(x)
        return np.mean(y_pred == y)


# Grid Search for optimal lambda_reg
lambda_values = [0.01, 0.1, 1, 10, 100]
best_lambda = None
best_score = 0

for lambda_reg in lambda_values:
    elm_ridge_model = ELM_Ridge(
        hidden_units=100, activation_function='sigmoid',
        x=X_train, y=y_train, lambda_reg=lambda_reg, elm_type='clf'
    )

    # Fit the model and get the training score
    beta, train_score, _ = elm_ridge_model.fit()

    if train_score > best_score:
        best_score = train_score
        best_lambda = lambda_reg

print(f"Best lambda_reg: {best_lambda} with training accuracy: {best_score * 100:.2f}%")

# Training the final model with the best lambda_reg
final_elm_ridge_model = ELM_Ridge(
    hidden_units=100, activation_function='sigmoid',
    x=X_train, y=y_train, lambda_reg=best_lambda, elm_type='clf'
)

beta, train_score, train_time = final_elm_ridge_model.fit()
print(f"Training Accuracy: {train_score * 100:.2f}%")
print(f"Training Time: {train_time:.4f} seconds")

# Test the model on the test set
test_score = final_elm_ridge_model.score(X_test, y_test)
print(f"Test Accuracy: {test_score * 100:.2f}%")

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.utils import shuffle

# Load dataset from CSV file
file_path = "/content/Scenario-B-merged_5s.csv"
df = pd.read_csv(file_path)

# Encode categorical 'label' column (if necessary)
df['label'] = df['label'].astype('category').cat.codes

# Drop non-numeric columns (e.g., IP addresses)

df = df.drop(columns=["Source IP", " Destination IP"], errors='ignore')
# Separate features and target variable
X = df.drop(columns=["label"]).values
y = df["label"].values

# Check for NaN or infinity values in X and y
if np.any(np.isnan(X)) or np.any(np.isinf(X)):
    print("Warning: NaN or Infinity values found in features. Replacing them with zero.")
    # Replace NaN and Inf values with 0
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

if np.any(np.isnan(y)) or np.any(np.isinf(y)):
    print("Warning: NaN or Infinity values found in target. Replacing them with zero.")
    # Replace NaN and Inf values in the target variable
    y = np.nan_to_num(y, nan=0, posinf=0, neginf=0)

# Feature Scaling (Standardization)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Feature Selection (Remove features with low variance)
selector = VarianceThreshold(threshold=0.01)  # Adjust threshold as needed
X = selector.fit_transform(X)

# Function to calculate the optimal lambda using cross-validation
def calculate_optimal_lambda_with_cv(X, y, k=5, lambda_values=None):
    if lambda_values is None:
        lambda_values = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]  # Default lambda values

    # Ensure X and y have the same number of samples
    n_samples = min(X.shape[0], y.shape[0])
    X = X[:n_samples]
    y = y[:n_samples]

    best_lambda = None
    best_r2_score = -float('inf')  # Initialize best score

    # KFold cross-validation
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    for lambda_reg in lambda_values:
        r2_scores = []  # List to store R^2 scores for each fold

        for train_index, val_index in kf.split(X):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            # Initialize and train the Ridge model with the current lambda value
            model = Ridge(alpha=lambda_reg)  # Using Ridge as a placeholder for REGELM
            model.fit(X_train, y_train)

            # Predict on validation set
            y_pred = model.predict(X_val)

            # Calculate R^2 score
            r2 = r2_score(y_val, y_pred)
            r2_scores.append(r2)

        # Calculate average R^2 score for this lambda
        avg_r2_score = np.mean(r2_scores)
        print(f"Lambda: {lambda_reg}, Average R^2 Score: {avg_r2_score}")

        # Update best lambda if current model performs better
        if avg_r2_score > best_r2_score:
            best_r2_score = avg_r2_score
            best_lambda = lambda_reg

    print(f"Best Lambda: {best_lambda}")
    return best_lambda

# Check class balance in 'y'
unique, counts = np.unique(y, return_counts=True)
print(f"Class balance in 'y': {dict(zip(unique, counts))}")

# Calculate the optimal lambda using cross-validation
best_lambda = calculate_optimal_lambda_with_cv(X, y)

# Print the best lambda value
print(f"The optimal lambda value for your REGELM model is: {best_lambda}")

import numpy as np
from scipy.linalg import pinv
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE
import time

# Load dataset
dataset = pd.read_csv("/content/Scenario-B-merged_5s.csv")

# Feature selection
targets = ['label']
feature_names = [col for col in dataset.columns if col not in targets]

# Handle categorical features
def cat_conv(data):
    data['Source IP'] = data['Source IP'].apply(hash).astype('float64')
    data[' Destination IP'] = data[' Destination IP'].apply(hash).astype('float64')
    return data

clean_data = cat_conv(dataset)

# Splitting data into X and y
def X_y_creation(dataset):
    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]
    return X, y

X, y_multi = X_y_creation(clean_data)

# Handling missing and infinite values
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.dropna(axis=0, inplace=True)
y_multi = y_multi.loc[X.index].reset_index(drop=True)

# Label Encoding
label_encoder = LabelEncoder()
y_multi_encoded = label_encoder.fit_transform(y_multi)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_multi_encoded, test_size=0.2, random_state=42, stratify=y_multi_encoded
)

# Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# SMOTE for Balancing Dataset
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Feature selection using ExtraTreesClassifier with hyperparameter tuning
clf = ExtraTreesClassifier(n_estimators=200, max_features='sqrt', random_state=42)
clf.fit(X_train, y_train)
model = SelectFromModel(clf, prefit=True)
X_train = model.transform(X_train)
X_test = model.transform(X_test)

# Ridge Regression-Based ELM
class ELM_Ridge:
    def __init__(self, hidden_units, activation_function, x, y, lambda_reg, elm_type, random_type='normal'):
        self.hidden_units = hidden_units
        self.activation_function = activation_function
        self.random_type = random_type
        self.x = x
        self.y = y
        self.class_num = len(np.unique(self.y))
        self.lambda_reg = lambda_reg
        self.elm_type = elm_type

        if self.elm_type == 'clf':
            self.encoder = LabelEncoder()
            self.y_encoded = self.encoder.fit_transform(self.y)
            self.y_temp = np.eye(self.class_num)[self.y_encoded]  # One-hot encoding

        if self.random_type == 'uniform':
            self.W = np.random.uniform(low=-1, high=1, size=(self.hidden_units, self.x.shape[1]))
            self.b = np.random.uniform(low=-1, high=1, size=(self.hidden_units, 1))
        else:  # normal distribution
            self.W = np.random.normal(loc=0, scale=0.5, size=(self.hidden_units, self.x.shape[1]))
            self.b = np.random.normal(loc=0, scale=0.5, size=(self.hidden_units, 1))

    def __input2hidden(self, x):
        H = np.dot(self.W, x.T) + self.b
        if self.activation_function == 'sigmoid':
            return 1 / (1 + np.exp(-H)), H
        elif self.activation_function == 'relu':
            return np.maximum(0, H), H
        elif self.activation_function == 'sin':
            return np.sin(H), H
        elif self.activation_function == 'tanh':
            return np.tanh(H), H
        elif self.activation_function == 'leaky_relu':
            return np.maximum(0, H) + 0.1 * np.minimum(0, H), H
        else:
            raise ValueError("Unsupported activation function")

    def __hidden2output(self, H):
        return np.dot(H.T, self.beta)

    def softmax(self, output):
        exp_values = np.exp(output - np.max(output, axis=1, keepdims=True))  # Stability trick (subtract max)
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)  # Normalize to get probabilities
        return probabilities

    def fit(self):
        start_time = time.time()
        H, _ = self.__input2hidden(self.x)

        # Ridge regression weight calculation
        H_t_H = np.dot(H, H.T) + self.lambda_reg * np.eye(H.shape[0])

        self.beta = np.dot(pinv(H_t_H), np.dot(H, self.y_temp))  # Using pseudo-inverse

        end_time = time.time()
        self.train_time = end_time - start_time

        train_output = self.__hidden2output(H)

        if self.elm_type == 'clf':
            predicted_labels = np.argmax(train_output, axis=1)
            self.train_score = np.mean(predicted_labels == self.y_encoded)

        return self.beta, self.train_score, self.train_time

    def predict(self, x):
        H, hidden_activations = self.__input2hidden(x)
        output = self.__hidden2output(H)

        # Calculate probabilities (assuming softmax activation for classification)
        probabilities = self.softmax(output)

        if self.elm_type == 'clf':
            return probabilities, H, hidden_activations, output
        return output

    def score(self, x, y):
        y_pred = self.predict(x)[0]  # Get probabilities for scoring
        y_pred_labels = np.argmax(y_pred, axis=1)  # Convert probabilities to class labels
        return np.mean(y_pred_labels == y)

# Main Execution
if __name__ == "__main__":
    elm_ridge_model = ELM_Ridge(
        hidden_units=500, activation_function='relu',
        x=X_train, y=y_train, lambda_reg=0.01, elm_type='clf'
    )

    beta, train_score, train_time = elm_ridge_model.fit()
    print(f"Training Accuracy: {train_score * 100:.2f}%")
    print(f"Training Time: {train_time:.4f} seconds")

    test_score = elm_ridge_model.score(X_test, y_test)
    print(f"Test Accuracy: {test_score * 100:.2f}%")

# Get the selected feature mask (True for selected features, False for unselected)
selected_features_mask = model.get_support()

# Get the original feature names
original_features = X.columns  # Assuming X is the original DataFrame before feature selection

# Get the names of selected features
selected_features = original_features[selected_features_mask]

# Print the selected features
print("Selected Features After Feature Selection:")
print(selected_features.tolist())
