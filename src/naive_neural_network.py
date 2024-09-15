"""
This module implements a two-layer neural network,
using Python Standard Library only.

It demonstrates the basics of neural networks and training with 
backpropagation without relying on any package.
"""

import random

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

# x is a 64x1000 matrix representing input data
x = [[random.gauss(0, 1) for j in range(D_in)] for i in range(N)]

# y is a 64x10 matrix representing target output data
y = [[random.gauss(0, 1) for j in range(D_out)] for i in range(N)]

# Initialize weights randomly.
# 'w1' connects the input layer to the hidden layer.
# 'w2' connects the hidden layer to the output layer.
# These weights will be updated during training to minimize the loss.

# w1 is a 1000x100 matrix of weights for input to hidden layer
w1 = [[random.gauss(0, 1) for j in range(H)] for i in range(D_in)]

# w2 is a 100x10 matrix of weights for hidden to output layer
w2 = [[random.gauss(0, 1) for j in range(D_out)] for i in range(H)]

def simple_relu(number: float) -> float:
    """
    Applies the ReLU activation function to a scalar value.

    Args:
    - y: A single numerical value.

    Returns:
    - The result of applying ReLU to y.
    """
    return max(0, number)

def matrix_multiply(matrix1: list[list],
                    matrix2: list[list]) -> list[list]:
    """
    Performs matrix multiplication on two 2D Python lists.

    Args:
    - matrix1: a 2D Python list
    - matrix2: a 2D Python list

    Returns:
    - A new 2D Python list as the result of the multiplication.
    """
    if len(matrix1[0]) != len(matrix2):
        raise ValueError("Number of columns in the first matrix must "
                         "be equal to the number of rows in the second matrix.")

    # Initialize the result matrix with zeros
    result = [[0] * len(matrix2[0]) for _ in range(len(matrix1))]

    # Perform matrix multiplication
    for i in range(len(matrix1)):
        for j in range(len(matrix2[0])):
            for k in range(len(matrix2)):
                result[i][j] += matrix1[i][k] * matrix2[k][j]

    return result

def relu(matrix: list[list]) -> list[list]:
    """
    Applies ReLU (Rectified Linear Unit) activation 
    function element-wise to a 2D matrix.

    Args:
    - matrix: a 2D Python list representing the matrix 
            to which ReLU will be applied.

    Returns:
    - A new 2D Python list with ReLU applied element-wise.
    """
    result = []
    for row in matrix:
        relu_row = [simple_relu(val) for val in row]
        result.append(relu_row)
    return result

def inplace_relu(matrix: list[list]) -> list[list]:
    """
    Applies ReLU (Rectified Linear Unit) activation function 
    element-wise to a 2D matrix in place.

    Args:
    - matrix: a 2D Python list representing 
              the matrix to which ReLU will be applied.

    Returns:
    - The input matrix with ReLU applied element-wise.
    """
    for row_idx in range(len(matrix)):
        for col_idx in range(len(matrix[0])):
            matrix[row_idx][col_idx] = max(0, matrix[row_idx][col_idx])

    return matrix

def transpose(matrix: list[list]) -> list[list]:
    """
    Transposes a 2D matrix (list of lists) in place.

    Args:
    - matrix: a 2D Python list representing the 
              matrix to be transposed.

    Returns:
    - A new 2D Python list that is the transpose of the input matrix.
    """
    # Get the number of rows and columns in the original matrix
    rows, cols = len(matrix), len(matrix[0])

    # Make a new matrix with dimensions swapped for transposition
    transposed = [[matrix[j][i] for j in range(rows)] for i in range(cols)]

    return transposed

def subtract_matrices(matrix1: list[list],
                      matrix2: list[list]) -> list[list]:
    """
    Subtract one 2D matrix from another element-wise.

    Args:
    - matrix1: a 2D Python list representing the first matrix.
    - matrix2: a 2D Python list representing the second matrix.

    Returns:
    - A new 2D Python list representing the result of the subtraction.
    """
    if len(matrix1) != len(matrix2) or len(matrix1[0]) != len(matrix2[0]):
        raise ValueError("Matrices must have the same dimensions")

    return [
        [matrix1[i][j] - matrix2[i][j] for j in range(len(matrix1[0]))]
        for i in range(len(matrix1))
    ]

def square_2d_list(matrix: list[list]) -> list[list]:
    """
    Computes the square of each element in a 2D Python list.

    Args:
    - matrix: a 2D Python list

    Returns:
    - A new 2D Python list with the squares of the input elements.
    """
    return [
        [x**2 for x in row]
        for row in matrix
    ]

def sum_2d_list(matrix: list[list]) -> float:
    """ 
    Computes the sum of each element in a 2D Python list.

    Args:
    - matrix: a 2D Python list

    Returns:
    - Sum of each element in 2D list."""
    res: float = 0.0
    for row in matrix:
        for elem in row:
            res += elem
    return res

def multiply_with_constant(matrix: list[list],
                           weight: float) -> list[list]:
    """ 
    Computes the weighted version of 2D Python list.

    Args:
    - matrix: a 2D Python list

    Returns:
    - The input matrix with each element multiplied by the constant.
    """
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            matrix[i][j] *= weight
    return matrix

def copy_2d_list(matrix: list[list]) -> list[list]:
    """
    Makes a copy of a given 2D Python list.

    Args:
    - matrix: a 2D Python list

    Returns:
    - A new 2D Python list containing the same elements as the input list.
    """
    return [row[:] for row in matrix]

learning_rate = 1e-6

for t in range(500):
    # ----------------------
    # Forward Pass
    # ----------------------

    # Forward pass: Compute predicted output.

    # Step 1: Compute the input to the hidden layer
    # by multiplying input 'x' with weights 'w1'.
    # This results in a matrix of shape (64, 100), representing
    # the activations of the hidden layer.
    h = matrix_multiply(x, w1)

    # Step 2: Apply the ReLU activation function.
    # ReLU (Rectified Linear Unit) introduces non-linearity
    # by setting all negative values to zero.
    # This helps the network learn complex patterns.
    h_relu = inplace_relu(h)  # Shape remains (64, 100)

    # Step 4: Compute the output by multiplying the activated
    # hidden layer 'h_relu' with weights 'w2'.
    # This results in the predicted output 'y_pred' of shape (64, 10).
    y_pred = matrix_multiply(h_relu, w2)

    # ----------------------
    # Compute Loss
    # ----------------------

    # Compute the loss: measure how far the predictions are
    # from the actual targets.
    # Here, we're using Mean Squared Error (MSE) loss.
    # (y_pred - y) computes the difference between predictions and targets.
    # square_2d_list(...) squares each difference to ensure it's positive.
    # sum_2d_list(...) adds up all the squared differences to get the total loss.
    loss = sum_2d_list(square_2d_list(subtract_matrices(y_pred, y)))

    # Print the current iteration and loss value.
    # This helps us monitor how the loss decreases over time
    # as the network learns.
    print(f"Current Iteration: {t} - Loss Value: {loss}")

    # ----------------------
    # Backpropagation
    # ----------------------

    # Gradient of loss with respect to y_pred.
    # Since loss = sum((y_pred - y)^2), the derivative is 2 * (y_pred - y).
    # Shape: (64, 10)
    grad_y_pred = multiply_with_constant(subtract_matrices(y_pred, y), 2.0)

    # Gradient of loss with respect to w2.
    # This is computed by multiplying the transpose
    # of 'h_relu' with 'grad_y_pred'.
    # The result is a matrix of shape (100, 10).
    # Shape: (100, 10)
    grad_w2 = matrix_multiply(transpose(h_relu), grad_y_pred)

    # Gradient of loss with respect to h_relu.
    # This is done by multiplying 'grad_y_pred' with the transpose of 'w2'.
    # The result is a matrix of shape (64, 100).
    # Shape: (64, 100)
    grad_h_relu = matrix_multiply(grad_y_pred, transpose(w2))

    # Make a copy of grad_h_relu to modify
    grad_h = copy_2d_list(grad_h_relu)

    # Apply the gradient mask based on where 'h_relu' was negative
    # Since ReLU sets all negative activations to zero,
    # the gradient should also be zero in those positions.
    grad_h = [
        [grad_h[i][j] if h_relu[i][j] >= 0 else 0.0 for j in range(H)]
        for i in range(N)
    ]

    # Gradient of loss with respect to w1.
    # This is computed by multiplying the transpose of 'x' with 'grad_h'.
    # The result is a matrix of shape (1000, 100).
    grad_w1 = matrix_multiply(transpose(x), grad_h)  # Shape: (1000, 100)

    # Compute gradients for biases
    grad_b2 = [0.0 for _ in range(D_out)]
    for i in range(N):
        for j in range(D_out):
            grad_b2[j] += grad_y_pred[i][j]

    grad_b1 = [0.0 for _ in range(H)]
    for i in range(N):
        for j in range(H):
            grad_b1[j] += grad_h[i][j]

    # ----------------------
    # Update Weights
    # ----------------------

    # Update weights using gradient descent
    w1 = subtract_matrices(w1, multiply_with_constant(grad_w1, learning_rate))
    w2 = subtract_matrices(w2, multiply_with_constant(grad_w2, learning_rate))
