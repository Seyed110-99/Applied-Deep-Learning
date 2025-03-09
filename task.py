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

def generate_multi_indices(D, M):
    """ Generate all multi-indices of degree less than or equal to M for D variables.
    Args:
        D: int, the number of variables.
        M: int, the maximum degree.
    Returns:
        multi_indices: list of tuples, each tuple is a multi-index.
    """
    
    multi_indices = []

    # Search for all possible combinations of exponents

    for m in range(M + 1):

        # Using combinations_with_replacement returns indices (positions) where the power is added.
        # For example, if D = 3 and m = 2, we may get (0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2).
        # We then convert these to exponent counts.

    
        combos = itertools.combinations_with_replacement(range(D), m)

        for comb in combos:
            
            # Initialize the exponents to zero
            exponents = [0] * D
            
            # Set the exponents to the values in comb
            for i in comb:
                exponents[i] += 1

            multi_indices.append(tuple(exponents))
            
    return torch.tensor(multi_indices, dtype=torch.float32)


def polynomial_features(x, M):
    """ Compute polynomial features of x up to degree M.
    Args:
        x: torch.Tensor of shape (D, ), the input data.
        M: int, the maximum degree.
    Returns:
        features: torch.Tensor of shape (N, K), the polynomial features.
    """
    # Vectors implementation

    # Get the number of samples and the number of variables
    D = x.shape[0]

    # Generate all multi-indices
    multi_indices = generate_multi_indices(D, M)

    # Compute the polynomial features
    
    features = torch.prod(torch.pow(x, multi_indices), dim=1)

    return features

def logistic_fun(w, M, x):
    """ Compute the logistic function.
    Args:
        w: torch.Tensor of shape (K,), the weights.
        M: int, the maximum degree.
        x: torch.Tensor of shape (D, ), the input data.
    Returns:
        y: torch.Tensor of shape (N,), the output.
    """
   # Compute the polynomial features
    features = polynomial_features(x, M)

    # Compute Z 
    z = torch.matmul(features, w)

    # Compute the output
    y = torch.sigmoid(z)

    return y

class MyCrossEntropy():
    def __init__(self):
        self.loss = None
        self.y_pred = None
        self.y_true = None
    
    def forward(self, y_pred, y_true):
        """ Compute the cross-entropy loss.
        Args:
            y_pred: torch.Tensor of Shape (N,)
            y_true: torch.Tensor of Shape (N,)
        Returns:
            loss: torch.Tensor of Shape (1)
        """
        self.y_pred = y_pred
        self.y_true = y_true
        self.loss = -torch.mean(y_true * torch.log(y_pred + 1e-7) + (1 - y_true) * torch.log(1 - y_pred + 1e-7))
        return self.loss

class MyRootMeanSquare():
    def __init__(self):
        self.loss = None
        self.y_pred = None
        self.y_true = None
    
    def forward(self, y_pred, y_true):
        """ Compute the root mean square loss.
        Args:
            y_pred: torch.Tensor of Shape (N,)
            y_true: torch.Tensor of Shape (N,)
        Returns:
            loss: torch.Tensor of Shape (1)
        """
        self.y_pred = y_pred
        self.y_true = y_true
        self.loss = torch.sqrt(torch.mean((y_pred - y_true) ** 2))
        return self.loss

def  fit_logistic_sgd(x, y_valid, M, loss_type = "CE", lr = 0.01, mini_batch_size =16, max_epochs=100):
    """ Fit a logistic reggression model using SGD.
    Args:
        x: torch.Tensor of shape (N, D), the input data.
        y_valid: torch.Tensor of shape (N,), the target values.
        M: int, the maximum degree.
        loss_type: str, the loss function to use. Either "CE" for cross-entropy or "RMS" for root mean square.
        lr: float, the learning rate.
        mini_batch_size: int, the mini-batch size.
        max_epochs: int, the maximum number of epochs.
    Returns:
        w: torch.Tensor of shape (M,), the weights.
    """

    N, D = x.shape

    # Get the number of features
    K = polynomial_features(x[0], M).shape[0]

    # Compute the polynomial features
    
    # Initialize the weights 
    weights = torch.randn(K, requires_grad=True)

    # Initialize the loss function
    if loss_type == "CE":
        loss_fn = MyCrossEntropy()
    elif loss_type == "RMS":
        loss_fn = MyRootMeanSquare()
    else:
        raise ValueError("The loss_type must be either 'CE' or 'RMS'.")

    # Initialise the optimizer
    optimizer = torch.optim.SGD([weights], lr=lr)

    # Training loop

    for epoch in range(max_epochs):

       # Shuffle the data

       indices = torch.randperm(N)

       # Loop over the mini-batches
       x_shuffled = x[indices]
       y_shuffled = y_valid[indices]

       for i in range(0, N, mini_batch_size):
           
            # Get the mini batch
            x_mini_batch = x_shuffled[i:i + mini_batch_size]
            y_mini_batch = y_shuffled[i:i + mini_batch_size]

            # Compute the output

            #y_pred = []

            # for x in x_mini_batch:
            #     y_pred_i = logistic_fun(weights, M, x)
            #     y_pred.append(y_pred_i)

            # # Convert the list to a tensor
            # y_pred = torch.stack(y_pred, type=torch.float32)

            #y_pred = torch.stack([logistic_fun(weights, M, x) for x in x_mini_batch])
            
            batched_logistic_fun = torch.vmap(logistic_fun, in_dims=(None, None, 0))
            y_pred = batched_logistic_fun(weights, M, x_mini_batch)

            # Compute the loss
            loss = loss_fn.forward(y_pred, y_mini_batch)

            if epoch == 0 or epoch == max_epochs - 1:
                print(f"Epoch {epoch}, Loss: {loss}")

            elif  epoch % 10 == 0 and not epoch == 0 and not epoch == max_epochs - 1: 
                print(f"Epoch {epoch}, Loss: {loss}")


            # Zero the gradients
            optimizer.zero_grad()

            # Compute the gradients
            loss.backward()

            # Update the weights
            optimizer.step()

    return weights

