"""This module implements a two layer neural network,
using PyTorch.

Let's see the basics of neural networks and 
training with backpropagation"""

import torch

# If you do not have a GPU, you can use
# device = torch.device('cpu')
device = torch.device('cuda')

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
# Input tensor of shape (64, 1000)
x = torch.randn(N, D_in, device=device)

# Target tensor of shape (64, 10)
y = torch.randn(N, D_out, device=device)


# Initialize weights randomly.
# 'w1' connects the input layer to the hidden layer.
# 'w2' connects the hidden layer to the output layer.
# These weights will be updated during training to minimize the loss.
w1 = torch.randn(D_in, H, device=device)
w2 = torch.randn(H, D_out, device=device)

# Set the learning rate, which determines how much the
# weights are updated during training.
learning_rate = 1e-6

for t in range(500):
    # ----------------------
    # Forward Pass
    # ----------------------

    # Compute the predicted y using the current weights.

    # Step 1: Compute the input to the hidden layer
    # by multiplying input 'x' with weights 'w1'.

    # Matrix multiplication of input and first layer weights
    h = x.mm(w1) # shape (64, 100)

    # Step 2: Apply the ReLU activation function.
    # ReLU (Rectified Linear Unit) introduces non-linearity
    # by setting all negative values to zero.

    # Applies ReLU; shape remains (64, 100)
    h_relu = h.clamp(min=0)

    # Step 3: Compute the output by multiplying the activated
    # hidden layer 'h_relu' with weights 'w2'.

    # Matrix multiplication with second layer weights; shape (64, 10)
    y_pred = h_relu.mm(w2)

    # ----------------------
    # Compute Loss
    # ----------------------

    # Compute the loss: measure how far the predictions
    # are from the actual targets.

    # Here, we're using Mean Squared Error (MSE) loss.
    # (y_pred - y) difference between predictions and targets.
    # .pow(2) squares each difference to ensure it's positive.
    # .sum() adds up all the squared differences to get the total loss.
    loss = (y_pred - y).pow(2).sum()

    # Print the current iteration and loss value.
    # This helps us monitor how the loss decreases
    # over time as the network learns.
    print(f"Current Iteration: {t} - Loss Value: {loss.item()}")

    # ----------------------
    # Backpropagation
    # ----------------------

    # Backpropagation: compute gradients of the loss with
    # respect to weights 'w1' and 'w2'.

    # These gradients indicate the direction and magnitude by
    # which we should adjust the weights to minimize the loss.

    # Gradient of loss with respect to y_pred.
    # Since loss = sum((y_pred - y)^2), the derivative is 2 * (y_pred - y).
    grad_y_pred = 2.0 * (y_pred - y)  # Shape: (64, 10)

    # Gradient of loss with respect to w2.
    # This is computed by multiplying the transpose
    # of 'h_relu' with 'grad_y_pred'.
    grad_w2 = h_relu.t().mm(grad_y_pred)  # Shape: (100, 10)

    # Gradient of loss with respect to h_relu.
    # This is done by multiplying 'grad_y_pred' with the transpose of 'w2'.
    grad_h_relu = grad_y_pred.mm(w2.t())  # Shape: (64, 100)

    # Gradient of loss with respect to h before the ReLU activation.
    # Since ReLU sets all negative values to zero, the gradient
    # should also be zero for those positions.
    grad_h = grad_h_relu.clone()  # Make a copy to avoid modifying 'grad_h_relu'
    grad_h[h < 0] = 0  # Apply the gradient mask based on where 'h' was negative

    # Gradient of loss with respect to w1.
    # This is computed by multiplying the transpose of 'x' with 'grad_h'.
    grad_w1 = x.t().mm(grad_h)  # Shape: (1000, 100)

    # ----------------------
    # Update Weights
    # ----------------------

    # Update the weights using gradient descent.
    # Subtract the gradient multiplied by the learning
    # rate from the current weights.
    # This moves the weights in the direction that most reduces the loss.
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
