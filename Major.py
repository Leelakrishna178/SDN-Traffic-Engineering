
import numpy as np
from scipy.linalg import pinv
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import time

# Load and preprocess data
raw_data = pd.read_csv("/content/Dataset.csv")

# Identify columns
label_col = ['label']
independent_cols = [feat for feat in raw_data.columns if feat not in label_col]

# Hash categorical IPs
def ip_hash_transform(frame):
    frame['Source IP'] = frame['Source IP'].apply(hash).astype('float64')
    frame[' Destination IP'] = frame[' Destination IP'].apply(hash).astype('float64')
    return frame

processed_df = ip_hash_transform(raw_data)

# Create features and labels
def separate_features_labels(df):
    features = df.iloc[:, :-1]
    targets = df.iloc[:, -1]
    return features, targets

features_matrix, target_vector = separate_features_labels(processed_df)

# Handle missing/infinite values
features_matrix.replace([np.inf, -np.inf], np.nan, inplace=True)
features_matrix.dropna(inplace=True)
target_vector = target_vector.loc[features_matrix.index].reset_index(drop=True)

# Encode output classes
target_encoder = LabelEncoder()
encoded_targets = target_encoder.fit_transform(target_vector)

# Split dataset
train_X, test_X, train_y, test_y = train_test_split(
    features_matrix, encoded_targets, test_size=0.2, random_state=42, stratify=encoded_targets
)

# Normalize features
zscore_scaler = StandardScaler()
train_X = zscore_scaler.fit_transform(train_X)
test_X = zscore_scaler.transform(test_X)

# Balance dataset using SMOTE
oversampler = SMOTE(random_state=42)
train_X, train_y = oversampler.fit_resample(train_X, train_y)

# Select top features
importance_model = ExtraTreesClassifier(n_estimators=200, max_features='sqrt', random_state=42)
importance_model.fit(train_X, train_y)
selected_features = SelectFromModel(importance_model, prefit=True)
train_X = selected_features.transform(train_X)
test_X = selected_features.transform(test_X)

# Ridge ELM model class
class RidgeELMClassifier:
    def __init__(self, hidden_size, activation, X_train, Y_train, lambda_param, mode, init_strategy='normal'):
        self.hidden_size = hidden_size
        self.activation = activation
        self.init_strategy = init_strategy
        self.X_train = X_train
        self.Y_train = Y_train
        self.total_classes = len(np.unique(Y_train))
        self.lambda_param = lambda_param
        self.mode = mode

        if self.mode == 'clf':
            self.label_mapper = LabelEncoder()
            self.numeric_labels = self.label_mapper.fit_transform(Y_train)
            self.one_hot_labels = np.eye(self.total_classes)[self.numeric_labels]

        if self.init_strategy == 'uniform':
            self.input_weights = np.random.uniform(-1, 1, (self.hidden_size, X_train.shape[1]))
            self.bias_vector = np.random.uniform(-1, 1, (self.hidden_size, 1))
        else:
            self.input_weights = np.random.normal(0, 0.5, (self.hidden_size, X_train.shape[1]))
            self.bias_vector = np.random.normal(0, 0.5, (self.hidden_size, 1))

    def _compute_hidden_layer(self, X_data):
        net_input = np.dot(self.input_weights, X_data.T) + self.bias_vector
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-net_input)), net_input
        elif self.activation == 'relu':
            return np.maximum(0, net_input), net_input
        elif self.activation == 'tanh':
            return np.tanh(net_input), net_input
        elif self.activation == 'sin':
            return np.sin(net_input), net_input
        elif self.activation == 'leaky_relu':
            return np.maximum(0, net_input) + 0.1 * np.minimum(0, net_input), net_input
        else:
            raise ValueError("Activation function not recognized")

    def _compute_output_layer(self, hidden_out):
        return np.dot(hidden_out.T, self.output_weights)

    def _apply_softmax(self, net_out):
        exp_vals = np.exp(net_out - np.max(net_out, axis=1, keepdims=True))
        return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)

    def train_model(self):
        start_stamp = time.time()
        hidden_out, _ = self._compute_hidden_layer(self.X_train)
        reg_matrix = np.dot(hidden_out, hidden_out.T) + self.lambda_param * np.identity(self.hidden_size)
        self.output_weights = np.dot(pinv(reg_matrix), np.dot(hidden_out, self.one_hot_labels))
        end_stamp = time.time()
        self.training_duration = end_stamp - start_stamp

        training_scores = self._compute_output_layer(hidden_out)
        if self.mode == 'clf':
            predicted_train_labels = np.argmax(training_scores, axis=1)
            self.training_accuracy = np.mean(predicted_train_labels == self.numeric_labels)

        return self.output_weights, self.training_accuracy, self.training_duration

    def generate_predictions(self, input_X):
        hidden_vals, activation_vals = self._compute_hidden_layer(input_X)
        net_scores = self._compute_output_layer(hidden_vals)
        final_probs = self._apply_softmax(net_scores)
        if self.mode == 'clf':
            return final_probs, hidden_vals, activation_vals, net_scores
        return net_scores

    def model_accuracy(self, X_eval, y_eval):
        pred_probs = self.generate_predictions(X_eval)[0]
        predicted_labels = np.argmax(pred_probs, axis=1)
        return np.mean(predicted_labels == y_eval)

# Main logic
if __name__ == "__main__":
    elm_net = RidgeELMClassifier(
        hidden_size=1200,
        activation='relu',
        X_train=train_X,
        Y_train=train_y,
        lambda_param=0.01,
        mode='clf'
    )

    trained_weights, acc_train, duration_train = elm_net.train_model()
    print(f"Training Accuracy: {acc_train * 100:.2f}%")
    print(f"Time Taken for Training: {duration_train:.4f} seconds")

    acc_test = elm_net.model_accuracy(test_X, test_y)
    print(f"Test Accuracy: {acc_test * 100:.2f}%")

    prob_preds, _, _, _ = elm_net.generate_predictions(test_X)
    class_preds = np.argmax(prob_preds, axis=1)

    # Evaluate performance
    conf_matrix = confusion_matrix(test_y, class_preds)
    print("Confusion Matrix:\n", conf_matrix)

    prec = precision_score(test_y, class_preds, average='weighted')
    rec = recall_score(test_y, class_preds, average='weighted')
    f1score = f1_score(test_y, class_preds, average='weighted')

    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1score:.4f}")
