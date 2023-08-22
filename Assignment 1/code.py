import csv
import random
import math
import numpy as np

def load_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append([float(val) for val in row])
    return data

def load_labels(file_path):
    labels = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            label = int(round(float(row[0])))
            labels.append([1 if i == label else 0 for i in range(4)])
    return labels

def normalize_data(data):
    max_val = max([max(row) for row in data])
    normalized_data = [[val / max_val for val in row] for row in data]
    return normalized_data

def k_fold_split(data, labels, k):
    combined_data = list(zip(data, labels))
    random.shuffle(combined_data)
    data, labels = zip(*combined_data)
    fold_size = len(data) // k
    folds = []
    for i in range(k):
        start = i * fold_size
        end = start + fold_size
        val_data = data[start:end]
        val_labels = labels[start:end]
        train_data = data[:start] + data[end:]
        train_labels = labels[:start] + labels[end:]
        folds.append((train_data, train_labels, val_data, val_labels))
    return folds

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

def initialize_weights(input_size, hidden_size, output_size):
    W1 = np.random.uniform(-1, 1, (input_size, hidden_size))
    b1 = np.random.uniform(-1, 1, hidden_size)
    W2 = np.random.uniform(-1, 1, (hidden_size, output_size))
    b2 = np.random.uniform(-1, 1, output_size)
    return W1, b1, W2, b2

def forward_propagation(X, W1, b1, W2, b2, activation_func):
    hidden_layer_input = np.dot(X, W1) + b1
    hidden_layer_output = activation_func(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, W2) + b2
    output_layer_output = softmax(output_layer_input)
    return hidden_layer_output, output_layer_output

def backward_propagation(X, y, W1, b1, W2, b2, hidden_layer_output, output_layer_output, learning_rate, activation_func):
    output_error = output_layer_output - y
    hidden_error = np.dot(output_error, W2.T) * activation_func(hidden_layer_output)

    dW2 = np.dot(hidden_layer_output.T, output_error)
    db2 = np.sum(output_error, axis=0)

    dW1 = np.dot(X.T, hidden_error)
    db1 = np.sum(hidden_error, axis=0)

    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    return W1, b1, W2, b2

def train_mlp(train_data, train_labels, hidden_size, learning_rate, num_epochs, activation_func, k):
    input_size = len(train_data[0])
    output_size = len(train_labels[0])
    folds = k_fold_split(train_data, train_labels, k)

    accuracies = []
    for fold in folds:
        train_data_fold, train_labels_fold, val_data_fold, val_labels_fold = fold
        W1, b1, W2, b2 = initialize_weights(input_size, hidden_size, output_size)

        for epoch in range(num_epochs):
            for X, y in zip(train_data_fold, train_labels_fold):
                X = np.array([X])
                y = np.array([y])

                hidden_layer_output, output_layer_output = forward_propagation(X, W1, b1, W2, b2, activation_func)
                W1, b1, W2, b2 = backward_propagation(X, y, W1, b1, W2, b2, hidden_layer_output, output_layer_output, learning_rate, activation_func)

        val_predictions = predict(val_data_fold, W1, b1, W2, b2, activation_func)
        val_accuracy = calculate_accuracy(val_predictions, val_labels_fold)
        accuracies.append(val_accuracy)

    avg_accuracy = sum(accuracies) / k
    return avg_accuracy

def predict(test_data, W1, b1, W2, b2, activation_func):
    predictions = []
    for X in test_data:
        X = np.array([X])
        _, output_layer_output = forward_propagation(X, W1, b1, W2, b2, activation_func)
        predicted_label = np.argmax(output_layer_output)
        predictions.append(predicted_label)
    return predictions

def calculate_accuracy(predictions, labels):
    correct = sum([1 for p, l in zip(predictions, labels) if p == np.argmax(l)])
    accuracy = correct / len(labels)
    return accuracy

# Load data and labels
data = load_data('train_data.csv')
labels = load_labels('train_labels.csv')

# Normalize data
data = normalize_data(data)

# Set hyperparameters
hidden_size = 64
learning_rate = 0.01
num_epochs = 1
# activation_func = relu  # Choose the activation function: sigmoid, relu, or tanh
k = 5

# Train and validate the MLP
accuracy = train_mlp(data, labels, hidden_size, learning_rate, num_epochs, relu, k)
print("Average Validation Accuracy by relu:", accuracy)

accuracy = train_mlp(data, labels, hidden_size, learning_rate, num_epochs, sigmoid, k)
print("Average Validation Accuracy by sigmoid:", accuracy)

accuracy = train_mlp(data, labels, hidden_size, learning_rate, num_epochs, tanh, k)
print("Average Validation Accuracy by tanh:", accuracy)