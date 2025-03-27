"""
GenAI usage statement:
This file was generated with assistive help from a generative AI tool to comment the code and suggest optimised methods for faster performance.
The original code was written by the author, and the AI tool provided suggestions for comments and structure.
"""
import torch
import itertools
from torch import Tensor
import math
from math import comb 
torch.manual_seed(42)  # Setting seed for reproducibility

# ---------------------------
# 1. Multi-index and Polynomial Features
# ---------------------------
def generate_multi_indices(D, M):
    """Generate all multi-indices of degree <= M for D variables.
    
    This function creates all possible exponent combinations for polynomial terms.
    For example, with D=2, M=2, the function generates indices for: 
    1, x1, x2, x1^2, x1*x2, x2^2
    
    Args:
        D (int): Number of variables.
        M (int): Maximum degree.
        
    Returns:
        torch.Tensor: Tensor of shape (K, D) containing all multi-indices.
    """
    multi_indices = []
    # Iterate over all possible values of m
    for m in range(M + 1):
        # Generate all combinations with replacement 
        combos = itertools.combinations_with_replacement(range(D), m)
        # Iterate over each combination and create a multi-index
        for comb in combos:
            exponents = [0] * D
            for i in comb:
                exponents[i] += 1
            multi_indices.append(tuple(exponents))
    return torch.tensor(multi_indices, dtype=torch.float32)

def polynomial_features(x, M):
    """Compute polynomial features of a single input vector x up to degree M.
    
    This transforms the input vector into a higher-dimensional feature space
    using polynomial combinations of the original features.
    
    Args:
        x (torch.Tensor): Input data of shape (D,).
        M (int): Maximum degree.
        
    Returns:
        torch.Tensor: Polynomial features of shape (K,), where K = sum_{m=0}^{M} binom(D+m-1, m).
    """
    # x is a 1-D tensor of shape (D,)
    D = x.shape[0]
    multi_indices = generate_multi_indices(D, M)  # shape: (K, D)
    # Expand x to shape (1, D) for broadcasting
    x_expanded = x.unsqueeze(0)  # shape: (1, D)
    # Compute elementwise power: result shape is (K, D)
    powered = torch.pow(x_expanded, multi_indices)  
    # Multiply along features to get a vector of shape (K,)
    features = torch.prod(powered, dim=1)
    return features

def logistic_fun(w, M, x):
    """Compute the logistic function for a single input vector.
    
    Implements σ(w·φ(x)) where φ(x) is the polynomial feature transformation
    and σ is the sigmoid function.
    
    Args:
        w (torch.Tensor): Weights of shape (K,), where K is the number of polynomial features.
        M (int): Maximum polynomial degree.
        x (torch.Tensor): Input vector of shape (D,).
        
    Returns:
        torch.Tensor: A scalar tensor representing the predicted probability.
    """
    # Compute polynomial features (returns a tensor of shape (K,))
    features = polynomial_features(x, M)
    # Compute the weighted sum (dot product)
    z = torch.dot(features, w)
    # Apply the sigmoid function
    y = torch.sigmoid(z)
    return y

# ---------------------------
# 2. Loss Classes
# ---------------------------
class MyCrossEntropy():
    """Cross-entropy loss implementation for binary classification.
    
    This loss function is commonly used for binary classification problems
    and measures the performance of a model that outputs probabilities.
    """
    def __init__(self):
        self.loss = None
    def forward(self, y_pred, y_true):
        """Compute cross-entropy loss.
        
        Args:
            y_pred (torch.Tensor): Predicted probabilities of shape (N,).
            y_true (torch.Tensor): True labels of shape (N,).
            
        Returns:
            torch.Tensor: Scalar loss.
        """
        # Add a small value to prevent log(0)
        y_pred = torch.clamp(y_pred, 0, 1)
        err = 1e-7  # Small constant to avoid numerical instability
        self.loss = -torch.mean(y_true * torch.log(y_pred + err) + (1 - y_true) * torch.log(1 - y_pred + err))
        return self.loss

class MyRootMeanSquare():
    """Root mean square error loss implementation.
    
    This loss function measures the square root of the average squared difference
    between predicted values and actual values.
    """
    def __init__(self):
        self.loss = None
    def forward(self, y_pred, y_true):
        """Compute RMSE loss.
        
        Args:
            y_pred (torch.Tensor): Predicted probabilities of shape (N,).
            y_true (torch.Tensor): True labels of shape (N,).
            
        Returns:
            torch.Tensor: Scalar loss.
        """
        y_pred = torch.clamp(y_pred, 0, 1)  # Ensure predictions are in [0,1]
        self.loss = torch.sqrt(torch.mean((y_pred - y_true) ** 2))
        return self.loss

# ---------------------------
# 3. SGD Training Function
# ---------------------------
def fit_logistic_sgd(x, y_valid, M, loss_type="CE", lr=0.01, mini_batch_size=10, max_epochs=100):
    """Fit a logistic regression model using SGD.
    
    Implements mini-batch stochastic gradient descent to find the optimal
    weight vector for the logistic regression model with polynomial features.
    
    Args:
        x (torch.Tensor): Input data of shape (N, D).
        y_valid (torch.Tensor): Target values of shape (N,).
        M (int): Maximum polynomial degree.
        loss_type (str): "CE" for cross-entropy or "RMS" for RMSE.
        lr (float): Learning rate.
        mini_batch_size (int): Mini-batch size.
        max_epochs (int): Number of epochs.
        
    Returns:
        torch.Tensor: Optimized weight vector of shape (K,).
    """
    N, D = x.shape
    # Determine the number of polynomial features (K) using the first sample
    K = polynomial_features(x[0], M).shape[0]
    # Initialize weights with shape (K,)
    weights = torch.randn(K, requires_grad=True)
    # Choose loss function
    if loss_type == "CE":
        loss_fn = MyCrossEntropy()
    elif loss_type == "RMS":
        loss_fn = MyRootMeanSquare()
    else:
        raise ValueError("loss_type must be 'CE' or 'RMS'.")
    # Set up optimizer
    optimizer = torch.optim.SGD([weights], lr=lr)
    
    for epoch in range(max_epochs):
        # Shuffle data for each epoch to ensure randomness in mini-batches
        indices = torch.randperm(N)
        x_shuffled = x[indices]
        y_shuffled = y_valid[indices]
    
        epoch_loss = 0.0
        num_batches = 0
        batch_accuracy = 0
        # Process data in mini-batches
        for i in range(0, N, mini_batch_size):
            # Zero the gradients before computing new gradients
            optimizer.zero_grad()

            x_mini_batch = x_shuffled[i:i+mini_batch_size]
            y_mini_batch = y_shuffled[i:i+mini_batch_size]
        
            # Apply logistic function to each sample in mini-batch
            # Using sequential processing instead of torch.vmap for compatibility
            y_pred = torch.stack([logistic_fun(weights, M, x) for x in x_mini_batch])
            
            # Compute loss for this mini-batch
            loss = loss_fn.forward(y_pred, y_mini_batch)
            
            # Backward pass and optimization step 
            loss.backward()
            optimizer.step()
            
            # Accumulate the loss over this mini-batch
            epoch_loss += loss.item()
            num_batches += 1
            batch_accuracy += compute_accuracy(y_pred, y_mini_batch)

        # Calculate accuracy and average loss for reporting
        batch_accuracy = batch_accuracy / num_batches
        epoch_loss /= num_batches
        
        # Print progress at regular intervals
        if epoch == 0 or epoch % 10 == 0 or epoch == max_epochs - 1:
            print(f"Epoch {epoch}, Avg Loss: {epoch_loss:.4f}, Accuracy: {batch_accuracy:.2%}")

    
    return weights

def compute_accuracy(y_pred, y_true):
    """Compute accuracy.
    
    Calculates the proportion of correctly classified samples.
    
    Args:
        y_pred (torch.Tensor): Predicted binary labels of shape (N,).
        y_true (torch.Tensor): True binary labels of shape (N,).
        
    Returns:
        float: Accuracy value.
    """
    N = y_pred.shape[0]
    y_pred = y_pred >= 0.5  # Convert probabilities to binary predictions
    accuracy = torch.sum(y_pred == y_true).float()/N
    return accuracy

def num_polynomial_terms(D, M):
    """
    Calculate the number of polynomial terms for D-dimensional input up to Mth order.
    
    The formula is: sum_{m=0..M} C(D+m-1, m), where C is the binomial coefficient.
    This represents the number of terms in a polynomial expansion with D variables up to degree M.
    """
    total = 0
    for m in range(M+1):
        total += comb(D + m - 1, m)
    return total

def generate_true_weights(D=5, M=2):
    """
    Generate the true weight vector w using the exact formula from the coursework:
      w_i = ((-1)^(p-i) * sqrt(p-i)) / p
    where p is the total number of polynomial terms.
    
    These weights define the ground truth model that generates the synthetic data.
    """
    p = num_polynomial_terms(D, M)
    w_list = []
    for i in range(p):
        exponent = p - i  # goes from p, p-1, ..., 1
        sign = (-1)**exponent
        value = sign * math.sqrt(exponent) / p
        w_list.append(value)
    return torch.tensor(w_list, dtype=torch.float32)

# ---------------------------
# 4. Main Script: Data Generation, Training, and Evaluation
# ---------------------------
if __name__ == "__main__":
    # Set parameters for the underlying true model and data generation
    M_true = 2      # Underlying true model uses M = 2
    D = 5           # Number of input dimensions
    
    # Determine number of polynomial features (K) for true model
    K = generate_multi_indices(D, M_true).shape[0]
    
    # Create true weight vector w_test using the specified formula.
    w_test = generate_true_weights(D, M_true)
    
    # Generate training and test data uniformly from [-5, 5]^5
    N_train = 200   # Number of training samples
    N_test = 100    # Number of test samples
    
    # Generate random input data within the specified range
    xTr = torch.empty((N_train, D)).uniform_(-5, 5)
    xTe = torch.empty((N_test, D)).uniform_(-5, 5)

    # Calculate true probabilities using the ground truth model
    yTr_true_prob = torch.stack([logistic_fun(w_test, M_true, x) for x in xTr])
    yTe_true_prob = torch.stack([logistic_fun(w_test, M_true, x) for x in xTe])
    
    
    # Add Gaussian noise with std=1.0 to create noisy data
    # Then threshold at 0.5 to generate binary targets
    yTr_noisy = yTr_true_prob + torch.normal(0, 1, (N_train,))
    yTr_noisy = (yTr_noisy >= 0.5).float()

    yTe_noisy = yTe_true_prob + torch.normal(0, 1, (N_test,))
    yTe_noisy = (yTe_noisy >= 0.5).float()

    # Generate clean binary labels (without noise) for comparison
    yTr_no_noise = (yTr_true_prob >= 0.5).float()
    yTe_no_noise = (yTe_true_prob >= 0.5).float()
    
    
    # ---------------------------
    # Training and Evaluation for each loss type and polynomial order M
    # ---------------------------
    # Dictionary to store accuracy results for different configurations
    losses = {"CE": {}, "RMS": {}}
    
    # Loop through different loss functions
    for loss_type in ["CE", "RMS"]:
        
        # Loop through different polynomial degrees
        for M in [1, 2, 3]:
            losses[loss_type][M] = []
            print(f"\nTraining with Loss: {loss_type}, Polynomial Order M: {M}")
            
            # Train the model with current configuration
            w_model = fit_logistic_sgd(xTr, yTr_noisy, M, loss_type=loss_type, lr=0.01, mini_batch_size=20, max_epochs=100)
            
            # Evaluate on training set
            yTr_pred_prob = torch.stack([logistic_fun(w_model, M, x) for x in xTr])
            yTr_pred = (yTr_pred_prob >= 0.5).float()
            
            # Calculate accuracies on training data (with and without noise)
            train_accuracy_noisy = compute_accuracy(yTr_pred, yTr_noisy)
            train_accuracy_no_noise = compute_accuracy(yTr_pred, yTr_no_noise)
            
            # Evaluate on test set
            yTe_pred_prob = torch.stack([logistic_fun(w_model, M, x) for x in xTe])
            yTe_pred = (yTe_pred_prob >= 0.5).float()
            
            # Calculate accuracies on test data (with and without noise)
            test_accuracy_noisy = compute_accuracy(yTe_pred, yTe_noisy)
            test_accuracy_no_noise = compute_accuracy(yTe_pred, yTe_no_noise)
            
            # Store all accuracy values (as percentages)
            losses[loss_type][M].append(int(round(float(train_accuracy_noisy) * 100)))
            losses[loss_type][M].append(int(round(float(test_accuracy_noisy) * 100)))
            losses[loss_type][M].append(int(round(float(train_accuracy_no_noise) * 100)))
            losses[loss_type][M].append(int(round(float(test_accuracy_no_noise) * 100)))


    # ---------------------------
    # Results Presentation and Analysis
    # ---------------------------

    print("\nAll Losses and Accuracies:")

    # Print results for no-noise accuracy with Cross-Entropy loss
    print("\nNo Noise dataset accuracy CE:")
    print(f"M: 1 train accuracy (yTraining no noise): {losses['CE'][1][2]}% and test accuracy (yTest no noise): {losses['CE'][1][3]}%")
    print(f"M: 2 train accuracy (yTraining no noise): {losses['CE'][2][2]}% and test accuracy (yTest no noise): {losses['CE'][2][3]}%")
    print(f"M: 3 train accuracy (yTraining no noise): {losses['CE'][3][2]}% and test accuracy (yTest no noise): {losses['CE'][3][3]}%")

    # Print results for no-noise accuracy with RMS loss
    print("\nNo Noise dataset accuracy RMS:")
    print(f"M: 1 train accuracy (yTraining no noise): {losses['RMS'][1][2]}% and test accuracy (yTest no noise): {losses['RMS'][1][3]}%")
    print(f"M: 2 train accuracy (yTraining no noise): {losses['RMS'][2][2]}% and test accuracy (yTest no noise): {losses['RMS'][2][3]}%")
    print(f"M: 3 train accuracy (yTraining no noise): {losses['RMS'][3][2]}% and test accuracy (yTest no noise): {losses['RMS'][3][3]}%")

    # Print results for noisy accuracy with Cross-Entropy loss
    print("\nNoisy dataset accuracy CE:")
    print(f"M: 1 train accuracy (yTraining noisy): {losses['CE'][1][0]}% and test accuracy (yTest noisy): {losses['CE'][1][1]}%")
    print(f"M: 2 train accuracy (yTraining noisy): {losses['CE'][2][0]}% and test accuracy (yTest noisy): {losses['CE'][2][1]}%")
    print(f"M: 3 train accuracy (yTraining noisy): {losses['CE'][3][0]}% and test accuracy (yTest noisy): {losses['CE'][3][1]}%")

    # Print results for noisy accuracy with RMS loss
    print("\nNoisy dataset accuracy RMS:")
    print(f"M: 1 train accuracy (yTraining noisy): {losses['RMS'][1][0]}% and test accuracy (yTest noisy): {losses['RMS'][1][1]}%")
    print(f"M: 2 train accuracy (yTraining noisy): {losses['RMS'][2][0]}% and test accuracy (yTest noisy): {losses['RMS'][2][1]}%")
    print(f"M: 3 train accuracy (yTraining noisy): {losses['RMS'][3][0]}% and test accuracy (yTest noisy): {losses['RMS'][3][1]}%")
    
    # Measure how much the noise affected the labels
    observed_train_accuracy_vs_true = compute_accuracy(yTr_noisy, yTr_no_noise)
    print(f"\nyTrain_noisy-vs-yTrain_no_noise = {observed_train_accuracy_vs_true*100:.1f}%")
    observed_test_accuracy_vs_true = compute_accuracy(yTe_noisy, yTe_no_noise)
    print(f"\nyTest_noisy-vs-yTst_no_noise = {observed_test_accuracy_vs_true*100:.1f}%")

    # ---------------------------
    # Metric justification and Result Analysis
    # ---------------------------
    print("\nAccuracy is chosen as the evaluation metric because it directly measures the proportion of correctly classified instances in this binary task, where each input must be assigned either 0 or 1. Its clarity, simplicity, and immediate interpretability make it ideal for evaluating logistic regression models that use probability thresholding.")
            
    print("\nExperimentally, the no-noise dataset accuracy indicates how well the model’s predictions match the true labels before noise, while the noisy dataset accuracy reflects the deviation introduced by noise. In our experiment (M=2), cross-entropy yields 69% accuracy on no-noise training data versus 64% with RMS, and similar trends are observed on ground truth test data (69% vs. 57%). Cross-entropy directly optimises class probabilities, making it more robust for binary classification, whereas RMS treats the task as regression and may overfit to noisy labels. Note that these results are experimental and may vary with different computers.")
