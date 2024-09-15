"""This script implements a two-layer neural network 
using NumPy.

It demonstrates the basics of neural networks and 
training with backpropagation without relying on 
deep learning frameworks like PyTorch or TensorFlow.
"""

# <3
import numpy as np

# Define the dimensions for the neural network layers:
# N: Number of samples per batch (batch size)
# D_in: Number of input features (dimensionality of input data)
# H: Number of neurons in the hidden layer
# D_out: Number of output features (dimensionality of output data)
N, D_in, H, D_out = 64, 1000, 100, 10

# Make random input and output data.
# In practice, 'x' would be your input data and 'y' would be
# the target labels.
# Here, we're using random data for demonstration purposes.

# Input matrix of shape (64, 1000)
x = np.random.randn(N, D_in)

# Target matrix of shape (64, 10)
y = np.random.randn(N, D_out)

# Initialize weights randomly.
# 'w1' connects the input layer to the hidden layer.
# 'w2' connects the hidden layer to the output layer.
# These weights will be updated during training to minimize the loss.
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

# Set the learning rate, which determines how much
# the weights are updated during training.
learning_rate = 1e-6

# Training loop: iterate 500 times to adjust the
# weights and train the network.
for t in range(500):
    # ----------------------
    # Forward Pass
    # ----------------------

    # Step 1: Compute the input to the hidden layer
    # by multiplying input 'x' with weights 'w1'.
    # This results in a matrix of shape (64, 100), representing
    # the activations of the hidden layer.
    h = x.dot(w1)

    # Step 2: Apply the ReLU activation function.
    # ReLU (Rectified Linear Unit) introduces non-linearity
    # by setting all negative values to zero.
    # This helps the network learn complex patterns.
    h_relu = np.maximum(h, 0)  # Shape remains (64, 100)

    # Step 3: Compute the output by multiplying the activated
    # hidden layer 'h_relu' with weights 'w2'.
    # This results in the predicted output 'y_pred' of shape (64, 10).
    y_pred = h_relu.dot(w2)

    # ----------------------
    # Compute Loss
    # ----------------------

    # Compute the loss: measure how far the predictions are
    # from the actual targets.
    # Here, we're using Mean Squared Error (MSE) loss.
    # (y_pred - y) computes the difference between predictions and targets.
    # np.square(...) squares each difference to ensure it's positive.
    # .sum() adds up all the squared differences to get the total loss.
    loss = np.square(y_pred - y).sum()

    # Print the current iteration and loss value.
    # This helps us monitor how the loss decreases over time
    # as the network learns.
    print(f"Current Iteration: {t} - Loss Value: {loss}")

    # ----------------------
    # Backpropagation
    # ----------------------

    # Backpropagation: compute gradients of the loss with
    # respect to weights 'w1' and 'w2'.
    # These gradients indicate the direction and magnitude
    # by which we should adjust the weights to minimize the loss.

    # Gradient of loss with respect to y_pred.
    # Since loss = sum((y_pred - y)^2), the derivative is 2 * (y_pred - y).
    grad_y_pred = 2.0 * (y_pred - y)  # Shape: (64, 10)

    # Gradient of loss with respect to w2.
    # This is computed by multiplying the transpose
    # of 'h_relu' with 'grad_y_pred'.
    # The result is a matrix of shape (100, 10).
    grad_w2 = h_relu.T.dot(grad_y_pred)  # Shape: (100, 10)

    # Gradient of loss with respect to h_relu.
    # This is done by multiplying 'grad_y_pred' with the transpose of 'w2'.
    # The result is a matrix of shape (64, 100).
    grad_h_relu = grad_y_pred.dot(w2.T)  # Shape: (64, 100)

    # Gradient of loss with respect to h before the ReLU activation.
    # Since ReLU sets all negative values to zero, the gradient
    # should also be zero for those positions.

    # Make a copy to avoid modifying 'grad_h_relu'
    grad_h = grad_h_relu.copy()

    # Apply the gradient mask based on where 'h' was negative
    grad_h[h < 0] = 0

    # Gradient of loss with respect to w1.
    # This is computed by multiplying the transpose of 'x' with 'grad_h'.
    # The result is a matrix of shape (1000, 100).
    grad_w1 = x.T.dot(grad_h)

    # ----------------------
    # Update Weights
    # ----------------------

    # Update the weights using gradient descent.
    # Subtract the gradient multiplied by the learning rate
    # from the current weights.
    # This moves the weights in the direction that most reduces the loss.
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
