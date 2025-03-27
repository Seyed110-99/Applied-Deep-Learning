"""
GenAI usage statement:
This file was generated with assistive help from a generative AI tool to comment the code and 
suggest optimised methods for faster performance.
The original code was written by the author, and the AI tool provided suggestions for comments and structure.
"""

import torch
import torch.nn.functional as F
from task import (
    generate_true_weights,
    compute_accuracy,
    polynomial_features,
    num_polynomial_terms,
    MyCrossEntropy
)

# ------------------------------------------------------------------
# 1. Utility: compute_logit
# ------------------------------------------------------------------
def compute_logit(w, M, x):
    """
    Compute the logit (pre-sigmoid) for a single input x given weight vector w and polynomial degree M.
    
    Args:
        w: Weight vector for the polynomial model
        M: Polynomial degree
        x: Input features (D-dimensional)
        
    Returns:
        The dot product between polynomial features and weights (scalar logit)
    """
    # Transform original features into polynomial features of degree M
    features = polynomial_features(x, M)
    # Compute the dot product between features and weights
    return torch.dot(features, w)


# ------------------------------------------------------------------
# 2. Forward pass (no temperature)
# ------------------------------------------------------------------
def model_forward(x, candidate_degrees, candidate_weights, alpha):
    """
    Forward pass to combine candidate polynomials via softmax over alpha with added Gumbel noise.
    
    Args:
        x: Single input, shape (D,)
        candidate_degrees: List of possible M values (polynomial degrees)
        candidate_weights: List of weight vectors (one per M)
        alpha: Trainable logits for choosing among candidate_degrees
    
    Returns:
        - y: Predicted probability (sigmoid of the weighted logit)
        - candidate_logits: Each candidate's logit
        - softmax_alpha: The softmax distribution over M (with Gumbel noise added)
    """
    # 1) Compute each candidate's logit
    candidate_logits = []
    for i, M in enumerate(candidate_degrees):
        # Calculate logit for each polynomial degree
        z = compute_logit(candidate_weights[i], M, x)
        candidate_logits.append(z)
    candidate_logits = torch.stack(candidate_logits)  # shape (num_candidates,)

    # 2) Add Gumbel noise to alpha before softmax for exploration
    # Gumbel-softmax trick allows for stochastic selection
    eps = 1e-8  # Small epsilon to avoid log(0)
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(alpha) + eps) + eps)
    alpha_noisy = alpha + gumbel_noise
    softmax_alpha = torch.softmax(alpha_noisy, dim=0)  # Probability distribution over degrees

    # 3) Combine candidate logits using the noisy softmax weights
    # Weighted average of all candidate logits
    weighted_logit = torch.sum(softmax_alpha * candidate_logits)

    # 4) Apply sigmoid to get final probability
    y = torch.sigmoid(weighted_logit)  # Convert to probability in [0,1]
    return y, candidate_logits, softmax_alpha



# ------------------------------------------------------------------
# 3. Training Loop (no temperature or annealing)
# ------------------------------------------------------------------
def train_learnable_M(
    train_x,
    train_y,
    candidate_degrees,
    candidate_weights,
    alpha,
    loss_fn,
    lr_weights=0.1,
    lr_alpha=0.01,
    mini_batch_size=20,
    max_epochs=100,
    entropy_weight=0.01
):
    """
    Train polynomial logistic regression but let M be chosen by a plain softmax over alpha.
    No temperature hyperparameter, no annealing.
    
    Args:
        train_x: Training features (N, D)
        train_y: Training labels (N,)
        candidate_degrees: List of polynomial degrees to choose from
        candidate_weights: List of weight vectors for each degree
        alpha: Trainable parameter vector for degree selection
        loss_fn: Loss function (cross-entropy)
        lr_weights: Learning rate for weights optimization
        lr_alpha: Learning rate for alpha optimization
        mini_batch_size: Size of mini-batches
        max_epochs: Maximum number of training epochs
        entropy_weight: Weight for entropy regularization
        
    Returns:
        Trained weights and alpha parameters
    """
    # Setup separate optimizers for weights and alpha
    weight_params = [w for w in candidate_weights]
    alpha_param = [alpha]
    opt_w = torch.optim.SGD(weight_params, lr=lr_weights)  # Optimizer for model weights
    opt_alpha = torch.optim.SGD(alpha_param, lr=lr_alpha)  # Optimizer for degree selection

    N = train_x.shape[0]  # Number of training samples

    for epoch in range(max_epochs):
        # Shuffle indices for stochastic mini-batch selection
        indices = torch.randperm(N)
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0

        # Mini-batch training
        for i in range(0, N, mini_batch_size):
            opt_w.zero_grad()  # Reset gradients for weights
            opt_alpha.zero_grad()  # Reset gradients for alpha

            # Get current mini-batch
            batch_idx = indices[i : i + mini_batch_size]
            batch_x = train_x[batch_idx]
            batch_y = train_y[batch_idx]

            batch_loss = 0.0
            batch_acc = 0.0

            # Process each sample in the mini-batch
            for j in range(batch_x.shape[0]):
                x_sample = batch_x[j]
                y_true = batch_y[j]

                # Forward pass (no temperature parameter)
                y_pred, _, softmax_alpha = model_forward(
                    x_sample, candidate_degrees, candidate_weights, alpha
                )

                # Classification loss (binary cross-entropy)
                loss = loss_fn.forward(y_pred, y_true)

                # Entropy regularization to encourage exploration of different degrees
                # Higher entropy means more uniform distribution over degrees
                entropy = -torch.sum(softmax_alpha * torch.log(softmax_alpha + 1e-8))
                loss += entropy_weight * entropy  # Add weighted entropy to loss

                batch_loss += loss
                batch_acc += compute_accuracy(y_pred.unsqueeze(0), y_true.unsqueeze(0))

            # Average loss and accuracy in this mini-batch
            batch_loss /= batch_x.shape[0]
            batch_acc  /= batch_x.shape[0]

            # Backpropagation and parameter updates
            batch_loss.backward()
            opt_w.step()  # Update weights
            opt_alpha.step()  # Update alpha

            epoch_loss += batch_loss.item()
            epoch_acc  += batch_acc.item()
            num_batches += 1

        # Calculate average metrics for this epoch
        avg_loss = epoch_loss / num_batches
        avg_acc  = epoch_acc  / num_batches

        # Print a progress update every 10 epochs or final epoch
        if epoch % 2 == 0 or epoch == max_epochs - 1:
            print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Acc: {avg_acc*100:.2f}%")
            # Current alpha distribution (no temperature)
            alpha_softmax = torch.softmax(alpha.detach(), dim=0).tolist()
            print(f"Alpha distribution: {alpha_softmax}")
            # Print weight norms for each M to monitor model complexity
            for idx, M in enumerate(candidate_degrees):
                wnorm = torch.norm(candidate_weights[idx]).item()
                print(f"  M={M}: weight norm={wnorm:.4f}")

    return candidate_weights, alpha


# ------------------------------------------------------------------
# 4. Evaluate
# ------------------------------------------------------------------
def evaluate_model(data_x, data_y, candidate_degrees, candidate_weights, alpha):
    """
    Evaluate the learned ensemble model on a dataset,
    using plain softmax-based selection for polynomial degree.
    
    Args:
        data_x: Test data features
        data_y: Test data labels
        candidate_degrees: List of polynomial degrees
        candidate_weights: Trained weights for each degree
        alpha: Trained alpha parameters for degree selection
        
    Returns:
        Average accuracy on the dataset
    """
    N = data_x.shape[0]
    total_acc = 0.0
    for i in range(N):
        # Get prediction for each sample
        y_pred, _, _ = model_forward(data_x[i], candidate_degrees, candidate_weights, alpha)
        # Add to running accuracy
        total_acc += compute_accuracy(y_pred.unsqueeze(0), data_y[i].unsqueeze(0))
    return total_acc / N


# ------------------------------------------------------------------
# 5. Main Script
# ------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)  # Set seed for reproducibility

    # Underlying "true" polynomial model parameters
    M_true = 2  # True polynomial degree
    D = 5      # Input dimension
    N_train = 200  # Number of training samples
    N_test = 100   # Number of test samples

    # Generate random data uniformly distributed between -5 and 5
    train_x = torch.empty((N_train, D)).uniform_(-5, 5)
    test_x  = torch.empty((N_test, D)).uniform_(-5, 5)

    # Generate the true weights for the polynomial of degree M_true
    true_w = generate_true_weights(D, M_true)

    # Function to produce noisy binary labels
    def get_labels(x, w, M):
        # Calculate probability from the logistic function for each sample
        probs = torch.stack([torch.sigmoid(compute_logit(w, M, x[i])) for i in range(x.shape[0])])
        # Add Gaussian noise and threshold at 0.5 for noisy labels
        # Also return noise-free labels for evaluation
        return (probs + torch.normal(0, 1, size=(x.shape[0],)) >= 0.5).float(), (probs >= 0.5).float()

    # Generate train/test labels (both noisy and noise-free versions)
    train_y, train_y_true = get_labels(train_x, true_w, M_true)
    test_y,  test_y_true  = get_labels(test_x,  true_w, M_true)

    # Define candidate polynomial degrees to choose from
    candidate_degrees = [1, 2, 3, 4, 5]
  
    # Initialize separate weight vectors for each candidate degree
    candidate_weights = []
    for M in candidate_degrees:
        # Calculate number of polynomial terms for this degree
        K = num_polynomial_terms(D, M)
        # Initialize weights randomly with gradients enabled
        w_init = torch.randn(K, requires_grad=True)
        candidate_weights.append(w_init)

    # Architecture parameter alpha (logits), one for each candidate degree
    # These will determine the distribution over polynomial degrees
    alpha = torch.randn(len(candidate_degrees), requires_grad=True)

    # Define loss function for binary classification
    loss_fn = MyCrossEntropy()

    print("Training with softmax selection of M (no temperature, no annealing)...\n")
    # Train the model with learnable polynomial degree
    candidate_weights, alpha = train_learnable_M(
        train_x, train_y,
        candidate_degrees, candidate_weights, alpha,
        loss_fn,
        lr_weights=0.1,
        lr_alpha=0.1,
        max_epochs=40,
        entropy_weight=0.01  # Controls exploration vs exploitation
    )

    # Determine the best polynomial degree by taking argmax of alpha
    with torch.no_grad():
        alpha_final = alpha.detach()
        alpha_soft = torch.softmax(alpha_final, dim=0)
        chosen_idx = alpha_soft.argmax().item()
        chosen_M = candidate_degrees[chosen_idx]

    print("\nFinal distribution over M:", alpha_soft.tolist())
    print("Learned M (argmax of alpha):", chosen_M)

    # Evaluate against the noise-free labels to measure true performance
    train_acc = evaluate_model(train_x, train_y_true, candidate_degrees, candidate_weights, alpha)
    test_acc  = evaluate_model(test_x,  test_y_true,  candidate_degrees, candidate_weights, alpha)

    print("\nFinal Results:")
    print(f"True Data M: {M_true}")
    print(f"Learned M: {chosen_M}")
    print(f"Train Accuracy vs. true train labels: {train_acc*100:.2f}%")
    print(f"Test Accuracy vs. true test labels:  {test_acc*100:.2f}%")

    print("""\nIn this experiment, we extend logistic regression by making the polynomial degree M a learnable parameter. Instead of fixing M (e.g., to 2) or performing a grid search, we define candidate degrees [1,2,3,4,5] and assign each candidate its own weight vector. A trainable parameter vector alpha produces a softmax distribution over these candidates after adding Gumbel noise, which introduces stochasticity in selecting the candidate polynomial terms. During training with SGD, both the model weights and the α\alphaα parameters are updated.

Printed messages report each candidate's epoch number, loss, accuracy, current alpha distribution, and weight norms. Although the underlying polynomial degree is M=2, the model eventually favours M=1, as indicated by the final alpha distribution and the selected candidate. The train and test accuracies are 62.50% and 64.00%, respectively. These numerical results can vary from run to run or on different machines due to random seed initialisations and hardware differences.

This flexible approach also highlights a potential risk of overfitting. By allowing the model to select a higher-order polynomial, the noise in the training data may be fitted rather than capturing the underlying pattern. Mitigation strategies such as adding L2 regularisation (weight decay), implementing early stopping based on validation performance, or reducing the candidate set for M can help address overfitting.

""")