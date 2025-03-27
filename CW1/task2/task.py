"""
GenAI usage statement:
This file was generated with assistive help from a generative AI tool to comment the code and suggest optimised methods for faster performance.
The original code was written by the author, and the AI tool provided suggestions for comments and structure.
All model definitions and utility functions are implemented using PyTorch.
"""

import time
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, RandomSampler
from PIL import Image, ImageDraw  # for drawing text on images
import numpy as np

# -------------------------
# Utility functions
# -------------------------
def compute_macro_f1(preds, labels, num_classes=10):
    """
    Compute the macro F1 score given the predicted labels and ground-truth labels.

    Args:
        preds (torch.Tensor): Tensor of predicted class indices with shape [N].
        labels (torch.Tensor): Tensor of ground truth class indices with shape [N].
        num_classes (int, optional): Number of classes. Default is 10.

    Returns:
        float: The macro F1 score averaged over all classes.
    """
    # Initialise list to store F1 scores for each class
    f1_scores = []
    # Iterate over each class index
    for c in range(num_classes):
        # Create boolean masks for predictions and ground truth for class c
        preds_c = (preds == c)
        labels_c = (labels == c)
        # Calculate true positives (correct predictions for class c)
        tp = (preds_c & labels_c).sum().item()
        # Calculate false positives (predictions of class c that are incorrect)
        fp = (preds_c & ~labels_c).sum().item()
        # Calculate false negatives (actual class c not predicted)
        fn = (~preds_c & labels_c).sum().item()
        # Calculate precision and recall; handle division by zero
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        # Compute F1 score or set to 0 if both precision and recall are zero
        f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
        # Append this F1 score to list
        f1_scores.append(f1)
    # Return average macro F1 score
    return sum(f1_scores) / num_classes

def evaluate_model(model, test_loader):
    """
    Evaluate the model on the test dataset and compute accuracy and macro F1.

    Args:
        model (torch.nn.Module): The trained model to evaluate.
        test_loader (DataLoader): DataLoader providing test data batches as (images, labels).

    Returns:
        tuple: (accuracy, macro_f1) where accuracy is a float (in %) and macro_f1 is the macro F1 score (in %).
    """
    model.eval()  # Set model to evaluation mode
    preds_list = []
    labels_list = []
    correct, total = 0, 0
    with torch.no_grad():  # No gradients needed for evaluation
        for images, labels in test_loader:
            outputs = model(images)  # Get model outputs
            preds = torch.argmax(outputs, dim=1)  # Predicted class indices
            preds_list.append(preds.cpu())
            # If labels are one-hot encoded, convert to indices
            if labels.ndimension() > 1:
                labels_cpu = labels.argmax(dim=1)
                labels_list.append(labels_cpu.cpu())
            else:
                labels_list.append(labels.cpu())
            if labels.ndimension() > 1:
                labels = labels.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    # Concatenate predictions and labels from all batches
    all_preds = torch.cat(preds_list)
    all_labels = torch.cat(labels_list)
    accuracy = (correct / total) * 100
    macro_f1 = compute_macro_f1(all_preds, all_labels, num_classes=10) * 100
    return accuracy, macro_f1

# -------------------------
# Model definitions
# -------------------------
class MyExtremeLearningMachine(nn.Module):
    """
    Extreme Learning Machine (ELM) model with one fixed convolutional layer and one trainable fully-connected layer.

    Methods:
        __init__: Initialize the ELM model.
        initialise_fixed_layers: Initialize the fixed convolutional layer weights.
        forward: Forward pass to compute class probabilities.
    """
    def __init__(self, num_feature_maps=16, kernel_size=3, num_classes=10, std_dev=0.01, seed=42):
        """
        Initialize the Extreme Learning Machine.

        Args:
            num_feature_maps (int): Number of output channels from the fixed convolutional layer.
            kernel_size (int): Size of the square convolutional kernel.
            num_classes (int): Number of output classes.
            std_dev (float): Standard deviation for initializing conv layer weights.
            seed (int): Random seed for reproducibility.
        """
        super(MyExtremeLearningMachine, self).__init__()
        torch.manual_seed(seed)  # Set seed for reproducibility
        # Define a fixed (non-trainable) convolutional layer
        self.conv_layer = nn.Conv2d(in_channels=3,
                                    out_channels=num_feature_maps,
                                    kernel_size=kernel_size,
                                    padding=kernel_size // 2,
                                    stride=1,
                                    bias=False)
        # Define a trainable fully-connected layer
        self.fc_layer = nn.Linear(num_feature_maps * 32 * 32, num_classes)
        # Freeze convolutional layer weights
        for param in self.conv_layer.parameters():
            param.requires_grad = False
        # Initialize the fixed layer weights
        self.initialise_fixed_layers(std_dev)
            
    def initialise_fixed_layers(self, std_dev):
        """
        Initialize the fixed convolutional layer weights with a Gaussian distribution.

        Args:
            std_dev (float): Standard deviation for weight initialization.
        """
        with torch.no_grad():
            self.conv_layer.weight.data.normal_(mean=0.0, std=std_dev)
            
    def forward(self, x):
        """
        Perform forward propagation through the ELM model.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 3, 32, 32].

        Returns:
            torch.Tensor: Output probabilities tensor of shape [batch_size, num_classes].
        """
        x = self.conv_layer(x)  # Convolution
        x = F.relu(x)           # ReLU activation
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layer(x)    # Fully-connected layer
        x = F.softmax(x, dim=1) # Softmax for probability distribution
        return x

def fit_elm_sgd(model, train_loader, num_epochs=5, lr=0.01, optimizer_type="SGD", mixup=False, checkpoints=None):
    """
    Train the ELM model using mini-batch gradient descent, optionally with MixUp augmentation.

    Args:
        model (torch.nn.Module): The ELM model to train.
        train_loader (DataLoader): DataLoader providing training batches as (images, labels).
        num_epochs (int): Number of training epochs.
        lr (float): Learning rate.
        optimizer_type (str): Type of optimizer to use ("SGD", "MomentumSGD", or "Adam").
        mixup (bool): Flag indicating whether to apply MixUp augmentation.
        checkpoints (list or None): List of epoch indices to save model checkpoints.

    Returns:
        tuple: (losses, accuracies) where losses is a list of average losses per epoch and accuracies is a list of epoch accuracies (in %).
    """
    if hasattr(model, 'models'):
        # If model is an ensemble, collect fc_layer parameters from each submodel
        all_fc_parameters = []
        for submodel in model.models:
            all_fc_parameters += list(submodel.fc_layer.parameters())
        optimizer_params = all_fc_parameters
    else:
        # Otherwise, use parameters of the single model's fc_layer
        optimizer_params = model.fc_layer.parameters()
    
    # Select optimizer type based on provided argument
    if optimizer_type == "SGD":
        optimizer = torch.optim.SGD(optimizer_params, lr=lr)
    elif optimizer_type == "MomentumSGD":
        optimizer = torch.optim.SGD(optimizer_params, lr=lr, momentum=0.9)
    elif optimizer_type == "Adam":
        optimizer = torch.optim.Adam(optimizer_params, lr=lr)
    else:
        raise ValueError("Invalid optimizer type")
    
    # Define loss function for classification using negative log likelihood
    criterion = nn.NLLLoss()
    losses, accuracies = [], []
    
    # Setup MixUp augmentation if enabled
    mixup_augment = MyMixUp(alpha=0.3, seed=42) if mixup else None
    if mixup_augment:
        mixup_augment.visualise_mixup(train_loader.dataset, num_images=16, save_path="mixup.png")
    
    start = time.time()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        correct, total = 0, 0
        epoch_preds = []
        epoch_labels = []
        for images, labels in train_loader:
            if mixup_augment:
                labels_one_hot = F.one_hot(labels, num_classes=10).float()
                images, mixed_labels = mixup_augment.mixup(images, labels_one_hot)
                outputs = model(images)
                loss = -torch.mean(torch.sum(mixed_labels * torch.log(outputs + 1e-10), dim=1))
                hard_labels = mixed_labels.argmax(dim=1)
            else:
                outputs = model(images)
                loss = criterion(torch.log(outputs), labels)
                hard_labels = labels

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == hard_labels).sum().item()
            total += hard_labels.size(0)
            epoch_preds.append(preds.cpu())
            epoch_labels.append(hard_labels.cpu())

        avg_loss = epoch_loss / len(train_loader)
        epoch_accuracy = (correct / total) * 100
        losses.append(avg_loss)
        accuracies.append(epoch_accuracy)
        all_epoch_preds = torch.cat(epoch_preds)
        all_epoch_labels = torch.cat(epoch_labels)
        epoch_f1 = compute_macro_f1(all_epoch_preds, all_epoch_labels, num_classes=10) * 100
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Accuracy = {epoch_accuracy:.2f}%, F1 = {epoch_f1:.2f}%")
        model.train()
        if checkpoints is not None and epoch in checkpoints:
            chkpt_name = f"checkpoint_ELM_Mixup_Epoch{epoch+1}.pt" if mixup else f"checkpoint_ELM_Epoch{epoch+1}.pt"
            torch.save(model.state_dict(), chkpt_name)
            print(f"Checkpoint saved to {chkpt_name}")
    end = time.time()
    print(f"Training with SGD took {(end - start)/60:.2f} minutes")
    return losses, accuracies

class MyMixUp():
    """
    Implements MixUp data augmentation for images and labels.

    Methods:
        __init__: Initialize the MixUp augmenter.
        _set_seed: Set the random seed.
        mixup: Create mixed images and labels.
        visualise_mixup: Generate and save a montage of mixed images.
    """
    def __init__(self, alpha=0.2, seed=42):
        """
        Initialize MixUp augmentation.

        Args:
            alpha (float): Alpha parameter for the Beta distribution.
            seed (int): Random seed for reproducibility.
        """
        self.alpha = alpha
        self.seed = seed
        self._set_seed(seed)
    
    def _set_seed(self, seed):
        """
        Set the random seed for PyTorch operations.

        Args:
            seed (int): Seed value.
        """
        torch.manual_seed(seed)
    
    def mixup(self, images, labels):
        """
        Generate mixed images and labels using a Beta-distributed mixing coefficient.

        Args:
            images (torch.Tensor): Batch of images with shape [batch_size, channels, height, width].
            labels (torch.Tensor): Batch of one-hot encoded labels with shape [batch_size, num_classes].

        Returns:
            tuple: (mixed_images, mixed_labels)
        """
        batch_size = images.size(0)
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample().item()
        indices = torch.randperm(batch_size)
        images_mixed = images[indices]
        labels_mixed = labels[indices]
        images_shuffled = lam * images + (1 - lam) * images_mixed
        labels_shuffled = lam * labels + (1 - lam) * labels_mixed
        return images_shuffled, labels_shuffled 
    
    def visualise_mixup(self, dataset, num_images=16, save_path="mixup.png"):
        """
        Create a montage of mixed images and save as a PNG file.

        Args:
            dataset (Dataset): Dataset object from which images and labels can be accessed.
            num_images (int): Number of images to include in the montage.
            save_path (str): Path to save the montage PNG.
        """
        self._set_seed(self.seed)
        indices = torch.randperm(len(dataset))
        images = torch.stack([dataset[i][0] for i in indices[:num_images]])
        labels = torch.tensor([dataset[i][1] for i in indices[:num_images]])
        one_hot_labels = F.one_hot(labels, num_classes=10).float()
        images_mixed, _ = self.mixup(images, one_hot_labels)
        grid = vutils.make_grid(images_mixed, nrow=int(num_images**0.5))
        ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        img = Image.fromarray(ndarr)
        img.save(save_path)
        print(f"MixUp visualization saved to {save_path}")

class MyEnsembleELM(nn.Module):
    """
    Ensemble of multiple Extreme Learning Machines (ELMs) for improved prediction.

    Methods:
        __init__: Initialize the ensemble with a specified number of ELM models.
        forward: Compute the average softmax output of all ensemble members.
        train_with_bootstrap: Train each ensemble member on a bootstrap sample of the training data.
    """
    def __init__(self, num_models=5, num_feature_maps=16, kernel_size=3, num_classes=10, std_dev=0.01, seed=42):
        """
        Initialize the ensemble of ELM models.

        Args:
            num_models (int): Number of ELM models in the ensemble.
            num_feature_maps (int): Number of feature maps per ELM model.
            kernel_size (int): Convolution kernel size.
            num_classes (int): Number of output classes.
            std_dev (float): Standard deviation for fixed layer weight initialization.
            seed (int): Base random seed.
        """
        super(MyEnsembleELM, self).__init__()
        torch.manual_seed(seed)
        if num_models < 1:
            raise ValueError("Number of models must be at least 1")
        if num_models > 20:
            warnings.warn("num_models is very high. This may slow down training.")
        if not (8 <= num_feature_maps <= 64):
            warnings.warn("num_feature_maps should be between 8 and 64 for efficient training.")
        if not (0.01 <= std_dev <= 0.5):
            warnings.warn("std_dev should be between 0.01 and 0.5 for effective weight initialization.")
        
        self.num_models = num_models
        models = []
        for i in range(num_models):
            model = MyExtremeLearningMachine(num_feature_maps=num_feature_maps,
                                             kernel_size=kernel_size,
                                             num_classes=num_classes,
                                             std_dev=std_dev, seed=seed + i)
            models.append(model)
        self.models = nn.ModuleList(models)
        
    def forward(self, x):
        """
        Forward pass: Compute the average softmax output from all ensemble models.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 3, 32, 32].

        Returns:
            torch.Tensor: Averaged prediction tensor of shape [batch_size, num_classes].
        """
        outputs = torch.stack([F.softmax(model(x), dim=1) for model in self.models])
        avg_outputs = torch.mean(outputs, dim=0)
        return avg_outputs

    def train_with_bootstrap(self, train_dataset, batch_size, num_epochs, lr, optimizer_type, mixup):
        """
        Train each ensemble member using bootstrap sampling.

        Args:
            train_dataset (Dataset): Training dataset.
            batch_size (int): Batch size for each DataLoader.
            num_epochs (int): Number of epochs for training.
            lr (float): Learning rate.
            optimizer_type (str): Optimizer type ("SGD", "MomentumSGD", or "Adam").
            mixup (bool): Whether to apply MixUp augmentation.

        Returns:
            None
        """
        for idx, model in enumerate(self.models):
            sampler = RandomSampler(train_dataset, replacement=True, num_samples=len(train_dataset))
            bootstrap_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=2)
            print(f"Training ensemble member {idx+1} with bootstrap sample...")
            fit_elm_sgd(model, bootstrap_loader, num_epochs=num_epochs, lr=lr, optimizer_type=optimizer_type, mixup=mixup)

# -------------------------
# Visualization
# -------------------------
def create_montage(model, test_loader, classes, save_path="result.png", num_images=36):
    """
    Create a montage of test images with ground-truth and predicted class labels.

    Args:
        model (torch.nn.Module): Trained model to use for prediction.
        test_loader (DataLoader): DataLoader for test data (images, labels).
        classes (list): List mapping class indices to class names.
        save_path (str): Path to save the montage PNG file.
        num_images (int): Number of images to include in the montage.

    Returns:
        None
    """
    model.eval()
    images_list = []
    labels_info = []
    count = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)
            for i in range(imgs.size(0)):
                if count >= num_images:
                    break
                img_tensor = imgs[i] * 0.5 + 0.5  # Unnormalize assuming CIFAR-10 normalization
                np_img = (img_tensor.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                pil_img = Image.fromarray(np_img)
                images_list.append(pil_img)
                gt_label = classes[labels[i].item()]
                pred_label = classes[preds[i].item()]
                labels_info.append((gt_label, pred_label))
                count += 1
            if count >= num_images:
                break

    grid_size = int(num_images ** 0.5)
    if grid_size * grid_size < num_images:
        grid_size += 1

    img_width, img_height = images_list[0].size
    montage = Image.new('RGB', (grid_size * img_width, grid_size * img_height))
    for idx, img in enumerate(images_list):
        row = idx // grid_size
        col = idx % grid_size
        montage.paste(img, (col * img_width, row * img_height))
    
    montage.save(save_path)
    print(f"\nMontage saved to {save_path}")
    print("\nOrdered list of ground-truth and predicted classes:")
    for idx, (gt, pred) in enumerate(labels_info, start=1):
        print(f"Image {idx}: GT: {gt}, Pred: {pred}")

# -------------------------
# Main execution (if run directly)
# -------------------------
if __name__ == "__main__":
    # Import training and testing functions from separate scripts
    from train import train_models
    from test import test_models
    print("Starting training process...")
    test_loader, classes, best_model_instance = train_models()
    print("Training complete. Starting testing process...")
    test_models()
    print("\nRandom guess definition and how it can be tested: In a multiclass classification, a random guess assigns each class with equal probability regardless of the input. For example, in a 10-class problem, each class is predicted with a 10% chance, yielding an expected accuracy of about 10%. To test this baseline, randomly assign a class label (uniformly sampled from all classes) to each test instance and compute metrics like accuracy and macro F1 score.\n")
    print(f"\nOn CIFAR-10, Accuracy shows how often the model is correct overall but may mask weaker performance on specific classes. Macro F1 ensures each of the 10 classes is treated equally by averaging per-class precision and recall. Together, they provide a robust measure of overall correctness and class-level fairness.")
    print(f"\nThe Plain ELM (no regularisation) improved from Epoch 1 (40.08% accuracy, 39.26% macro F1) to Epoch 3 (42.85%, 42.07%) and finally reached 43.61% accuracy, 42.58% macro F1 at the last epoch. Incorporating MixUp started lower at Epoch 1 (40.53% accuracy, 39.98% macro F1) but climbed steadily, ending at 43.65% accuracy, 42.75% macro F1—indicating a modest yet consistent regularisation benefit. The Ensemble ELM (bootstrap sampling) saw a marked jump even with one model (42.23% accuracy, 40.68% F1), rising to 46.85% accuracy, 46.14% F1 with three models, and peaking at 47.28% accuracy, 46.46% F1 with the full ensemble of five. In contrast, the Ensemble ELM with MixUp began at 43.23% accuracy, 42.07% F1 (one model), improved to 45.73% accuracy, 44.82% F1 (three models), and reached 45.97% accuracy, 45.15% F1 at full capacity. Interestingly, ensembling alone outperformed ensembling plus MixUp, possibly due to over-regularisation when combining both techniques. These results are experimental and may vary across different runs or hardware configurations. Overall, ensembling provides the largest performance gains, while MixUp adds a smaller but valuable improvement in model generalisation.\n")
