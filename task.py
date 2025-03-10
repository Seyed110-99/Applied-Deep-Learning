#def polynomial_features(x, M):
#     """ Compute polynomial features of x up to degree M.
#     Args:
#         x: torch.Tensor of shape (N, D), the input data.
#         M: int, the maximum degree.
#     Returns:
#         features: torch.Tensor of shape (N, K), the polynomial features.
#     """ 
# ###########################################################

#     # Matrix implementation

#     D = x.shape[1]

#     # Generate all multi-indices
#     multi_indices = generate_multi_indices(D, M)

#     # Compute the polynomial features

#     # Add a dimension to x (N, 1, D)
#     x_expanded = x.unsqueeze(1)

#     # Add a dimension to multi_indices (1, K, D)
#     multi_indices_expanded = multi_indices.unsqueeze(0)

#     # Compute the polynomial features (N, K, D) and take the product along the last dimension
#     features = torch.prod(torch.pow(x_expanded, multi_indices_expanded), dim=2)

#     return features

import torch
import itertools

# ---------------------------
# 1. Multi-index and Polynomial Features
# ---------------------------
def generate_multi_indices(D, M):
    """Generate all multi-indices of degree <= M for D variables.
    
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
        self.loss = -torch.mean(y_true * torch.log(y_pred + 1e-7) + (1 - y_true) * torch.log(1 - y_pred + 1e-7))
        return self.loss

class MyRootMeanSquare():
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
        self.loss = torch.sqrt(torch.mean((y_pred - y_true) ** 2))
        return self.loss

# ---------------------------
# 3. SGD Training Function
# ---------------------------
def fit_logistic_sgd(x, y_valid, M, loss_type="CE", lr=0.01, mini_batch_size=16, max_epochs=100):
    """Fit a logistic regression model using SGD.
    
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
        # Shuffle data
        indices = torch.randperm(N)
        x_shuffled = x[indices]
        y_shuffled = y_valid[indices]
    
        epoch_loss = 0.0
        num_batches = 0

        for i in range(0, N, mini_batch_size):
            x_mini_batch = x_shuffled[i:i+mini_batch_size]
            y_mini_batch = y_shuffled[i:i+mini_batch_size]
        
            batched_logistic_fun = torch.vmap(logistic_fun, in_dims=(None, None, 0))
            y_pred = batched_logistic_fun(weights, M, x_mini_batch)
        
            loss = loss_fn.forward(y_pred, y_mini_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate the loss over this mini-batch
            epoch_loss += loss.item()
            num_batches += 1

        # Compute the average loss for this epoch
        epoch_loss /= num_batches

        # Print the *average* epoch loss
        if epoch == 0 or epoch % 10 == 0 or epoch == max_epochs - 1:
            print(f"Epoch {epoch}, Avg Loss: {epoch_loss:.4f}")

    
    return weights

def compute_accuracy(y_pred, y_true):
    """Compute accuracy.
    
    Args:
        y_pred (torch.Tensor): Predicted binary labels of shape (N,).
        y_true (torch.Tensor): True binary labels of shape (N,).
        
    Returns:
        float: Accuracy value.
    """
    N = y_pred.shape[0]
    accuracy = torch.sum(y_pred == y_true).float() / N
    return accuracy

# ---------------------------
# 4. Main Script: Data Generation, Training, and Evaluation
# ---------------------------
if __name__ == "__main__":
    # Set parameters for the underlying true model and data generation
    M_true = 2      # Underlying true model uses M = 2
    D = 5
    # Determine number of polynomial features (K) for true model
    K = generate_multi_indices(D, M_true).shape[0]
    # Create true weight vector w_test using the specified formula.
    # Example formula: w_i = (-1)^(K-i) * (sqrt(K-i) / K)
    w_test = torch.stack([((-1)**i) * (torch.sqrt(torch.tensor(i, dtype=torch.float32)) / K) 
                           for i in range(K, 0, -1)])
    
    # Generate training and test data uniformly from [-5, 5]^5
    N_train = 200
    N_test = 100
    xTr = (torch.rand(N_train, D) * 10) - 5
    xTe = (torch.rand(N_test, D) * 10) - 5
    
    # Use vmap to compute true probabilities for generated data
    batched_logistic_fun = torch.vmap(logistic_fun, in_dims=(None, None, 0))
    yTr_true_prob = batched_logistic_fun(w_test, M_true, xTr)
    yTe_true_prob = batched_logistic_fun(w_test, M_true, xTe)
    
    # Add Gaussian noise with std=1.0 and threshold at 0.5 to generate binary targets
    yTr_noisy = yTr_true_prob + torch.randn(N_train) * 1.0
    yTe_noisy = yTe_true_prob + torch.randn(N_test) * 1.0
    yTr = (yTr_noisy >= 0.5).float()
    yTe = (yTe_noisy >= 0.5).float()
    
    # ---------------------------
    # Training and Evaluation for each loss type and polynomial order M
    # ---------------------------
    losses = {"CE": {}, "RMS": {}}
    for loss_type in ["CE", "RMS"]:
        for M in [1, 2, 3]:
            losses[loss_type][M] = []
            print(f"\nTraining with Loss: {loss_type}, Polynomial Order M: {M}")
            w_model = fit_logistic_sgd(xTr, yTr, M, loss_type=loss_type, lr=0.1, mini_batch_size=16, max_epochs=200)
            
            # Evaluate on training set
            yTr_pred_prob = torch.vmap(logistic_fun, in_dims=(None, None, 0))(w_model, M, xTr)
            yTr_pred = (yTr_pred_prob >= 0.5).float()
            train_accuracy = compute_accuracy(yTr_pred, yTr)
            
            # Evaluate on test set
            yTe_pred_prob = torch.vmap(logistic_fun, in_dims=(None, None, 0))(w_model, M, xTe)
            yTe_pred = (yTe_pred_prob >= 0.5).float()
            test_accuracy = compute_accuracy(yTe_pred, yTe)
            
            losses[loss_type][M].append((int(round(float(train_accuracy) * 100)), int(round(float(test_accuracy) * 100))))
            # print(f"Loss type: {loss_type}, M: {M}")
            # print(f"Training Accuracy: {train_accuracy:.2%}, Test Accuracy: {test_accuracy:.2%}")
    
    print("\nAll Losses and Accuracies:")
    print(losses)
    
    # # ---------------------------
    # # Metric justification (<=50 words)
    # # ---------------------------
    # print("\nMetric Justification: Accuracy is chosen because it directly measures the proportion of correctly classified samples, which is vital for evaluating binary classification performance.")
    
    # # ---------------------------
    # # Loss Function Comparison (<=100 words)
    # # ---------------------------
    # print("\nLoss Function Comparison: Cross-entropy loss tends to produce steeper gradients for misclassified samples, leading to faster convergence and more decisive predictions. In contrast, RMSE loss provides a smoother gradient but can result in slower convergence. Our experiments indicate that cross-entropy generally achieves higher accuracy on the test set compared to RMSE, reflecting its suitability for classification.")

    # # Note: Extend these experiments in task1a.py to make M a learnable parameter.
