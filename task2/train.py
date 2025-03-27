"""
GenAI usage statement:
This file was generated with assistive help from a generative AI tool to comment the code and suggest optimised methods for faster performance.
The original code was written by the user, and the AI tool provided suggestions for comments and structure.

train.py

This module defines the function train_models() which performs all training routines for Task 2.
It loads the data, trains four models:
  1. Plain ELM (no regularisation)
  2. ELM with MixUp augmentation
  3. Ensemble of ELM models (with bootstrap sampling)
  4. Ensemble of ELM models with MixUp
The trained models are saved to disk and the best model is selected based on the composite score.
"""

import torch
from torch.utils.data import DataLoader, RandomSampler
from torchvision import datasets, transforms
from task import (MyExtremeLearningMachine, MyEnsembleELM, fit_elm_sgd, evaluate_model, create_montage)
# Import necessary packages for data preparation and model training
def train_models():
    """
    Main function that handles the complete model training pipeline for Task 2.
    
    This function:
    1. Prepares and loads the CIFAR-10 dataset
    2. Trains four different models:
       - Plain ELM with no regularization
       - ELM with MixUp data augmentation
       - Ensemble of ELM models using bootstrap sampling
       - Ensemble of ELM models with both bootstrap sampling and MixUp
    3. Evaluates each model on the test set
    4. Saves all models to disk
    5. Selects the best model based on a composite score of accuracy and F1 score
    
    Returns:
        tuple: (testloader, classes, best_model_instance) containing the test data loader,
               class names, and the best performing model instance
    """
    # -------------------------
    # Data Preparation
    # -------------------------
    # Standard normalization transform for CIFAR-10 images
    transform = transforms.Compose([
        transforms.ToTensor(),                               # Convert images to PyTorch tensors
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))  # Normalize with mean=0.5, std=0.5 for each channel
    ])
    batch_size = 32  # Standard batch size for training
    epochs = 5       # Number of training epochs for each model
    
    # Load CIFAR-10 training data
    train_dataset = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # Load CIFAR-10 test data
    test_dataset = datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # CIFAR-10 class names
    classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    
    # Print dataset information for verification
    print(f"Train Dataset: {len(train_dataset)} images")
    print(f"Test Dataset: {len(test_dataset)} images")
    image, label = train_dataset[0]
    print(f"Image Shape: {image.shape}")
    print(f"Label: {label}")

    # -------------------------
    # Training Plain ELM
    # -------------------------
    # Initialize the standard ELM model without any regularization techniques
    model_ELM = MyExtremeLearningMachine(num_feature_maps=8, kernel_size=3, num_classes=10, std_dev=0.05)
    print("\nStarting Training... without regularisation")
    # Train the model using stochastic gradient descent with momentum
    losses, train_accuracies = fit_elm_sgd(model_ELM, trainloader, num_epochs=epochs, lr=0.01,
                                           optimizer_type="MomentumSGD", mixup=False, checkpoints=[0,2])
    # Evaluate the trained model on the test set
    test_accuracy, f1_elm = evaluate_model(model_ELM, testloader)
    print(f"\nPlain ELM - Test Accuracy: {test_accuracy:.2f}%, Macro F1: {f1_elm:.2f}%")

    # -------------------------
    # Training ELM with MixUp
    # -------------------------
    # Initialize another ELM model that will be trained with MixUp augmentation
    model_ELM_mixup = MyExtremeLearningMachine(num_feature_maps=8, kernel_size=3, num_classes=10, std_dev=0.05)
    print("\nStarting Training... with MixUp augmentation")
    # Train using MixUp data augmentation technique (mixup=True)
    losses_mixup, train_accuracies_mixup = fit_elm_sgd(model_ELM_mixup, trainloader, num_epochs=epochs, lr=0.01,
                                                     optimizer_type="MomentumSGD", mixup=True, checkpoints=[0,2])
    # Evaluate the MixUp-trained model
    test_accuracy_mixup, f1_elm_mixup = evaluate_model(model_ELM_mixup, testloader)
    print(f"\nELM with MixUp - Test Accuracy: {test_accuracy_mixup:.2f}%, Macro F1: {f1_elm_mixup:.2f}%")

    # -------------------------
    # Training Ensemble of ELM Models (Bootstrap Sampling)
    # -------------------------
    # Initialize an ensemble of 5 ELM models
    ensemble_model = MyEnsembleELM(num_models=5 , num_feature_maps=8, kernel_size=3, num_classes=10, std_dev=0.05, seed=42)
    print("\nStarting Training... Ensemble of ELM models with bootstrap sampling")
    # Train each model in the ensemble using bootstrap sampling (random sampling with replacement)
    for idx, model in enumerate(ensemble_model.models):
        # Create a bootstrap sample by randomly sampling with replacement
        sampler = RandomSampler(train_dataset, replacement=True, num_samples=len(train_dataset))
        bootstrap_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=2)
        print(f"\nTraining ensemble member {idx+1} with bootstrap sample...")
        # Train the individual model on its bootstrap sample
        fit_elm_sgd(model, bootstrap_loader, num_epochs=epochs, lr=0.01, optimizer_type="MomentumSGD", mixup=False)
        # Save intermediate checkpoints during ensemble training
        if idx + 1 == 1 or idx + 1 == 3:
            checkpoint = {"ensemble_state": [m.state_dict() for m in ensemble_model.models[:idx+1]],
                          "num_models": idx + 1}
            torch.save(checkpoint, f"ensemble_elm_check{idx+1}.pt")
            print(f"Checkpoint saved: ensemble with {idx+1} models.")
    # Evaluate the full ensemble model
    test_accuracy_ensemble, f1_ensemble = evaluate_model(ensemble_model, testloader)
    print(f"\nEnsemble ELM - Test Accuracy: {test_accuracy_ensemble:.2f}%, Macro F1: {f1_ensemble:.2f}%")

    # -------------------------
    # Training Ensemble of ELM Models with MixUp and Bootstrap Sampling
    # -------------------------
    # Initialize another ensemble, this time for training with both bootstrap sampling and MixUp
    ensemble_model_mixup = MyEnsembleELM(num_models=5 , num_feature_maps=8, kernel_size=3, num_classes=10, std_dev=0.05, seed=42)
    print("\nStarting Training... Ensemble of ELM models with MixUp and bootstrap sampling")
    # Train each model with both bootstrap sampling and MixUp augmentation
    for idx, model in enumerate(ensemble_model_mixup.models):
        # Create a bootstrap sample
        sampler = RandomSampler(train_dataset, replacement=True, num_samples=len(train_dataset))
        bootstrap_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=2)
        print(f"\nTraining ensemble member {idx+1} with bootstrap sample...")
        # Train with MixUp enabled (mixup=True)
        fit_elm_sgd(model, bootstrap_loader, num_epochs=epochs, lr=0.01, optimizer_type="MomentumSGD", mixup=True)
        # Save intermediate checkpoints
        if idx + 1 == 1 or idx + 1 == 3:
            checkpoint = {"ensemble_state": [m.state_dict() for m in ensemble_model_mixup.models[:idx+1]],
                          "num_models": idx + 1}
            torch.save(checkpoint, f"ensemble_elm_mixup_check{idx+1}.pt")
            print(f"Checkpoint saved: ensemble mixup with {idx+1} models.")
    # Evaluate the ensemble with MixUp
    test_accuracy_ensemble_mixup, f1_ensemble_mixup = evaluate_model(ensemble_model_mixup, testloader)
    print(f"\nEnsemble with MixUp - Test Accuracy: {test_accuracy_ensemble_mixup:.2f}%, Macro F1: {f1_ensemble_mixup:.2f}%")

    # -------------------------
    # Saving Trained Models
    # -------------------------
    # Save all four trained models to disk for later use
    torch.save(model_ELM.state_dict(), "elm.pt")
    torch.save(model_ELM_mixup.state_dict(), "elm_mixup.pt")
    torch.save(ensemble_model.state_dict(), "ensemble_elm.pt")
    torch.save(ensemble_model_mixup.state_dict(), "ensemble_elm_mixup.pt")
    print("Models saved to disk")

    # -------------------------
    # Best Model Selection
    # -------------------------
    # Calculate composite scores (average of accuracy and F1) for each model
    score_elm = (test_accuracy + f1_elm) / 2
    score_mixup = (test_accuracy_mixup + f1_elm_mixup) / 2
    score_ensemble = (test_accuracy_ensemble + f1_ensemble) / 2
    score_ensemble_mixup = (test_accuracy_ensemble_mixup + f1_ensemble_mixup) / 2
    
    # Store all scores in a dictionary for comparison
    scores = {
        "ELM": score_elm,
        "ELM with MixUp": score_mixup,
        "Ensemble of ELM models": score_ensemble,
        "Ensemble of ELM models with MixUp": score_ensemble_mixup
    }
    
    # Find the model with the highest composite score
    best_model_name = max(scores, key=scores.get)
    best_score = scores[best_model_name]
    # Select the corresponding model instance
    if best_model_name == "ELM":
        best_model_instance = model_ELM
    elif best_model_name == "ELM with MixUp":
        best_model_instance = model_ELM_mixup
    elif best_model_name == "Ensemble of ELM models":
        best_model_instance = ensemble_model
    elif best_model_name == "Ensemble of ELM models with MixUp":
        best_model_instance = ensemble_model_mixup

    print(f"\nBest model: {best_model_name} with composite score (Accuracy+F1)/2: {best_score:.2f}")

    # -------------------------
    # Visualize results from the best model (here we load the plain ELM for demonstration)
    # -------------------------
    best_model = MyExtremeLearningMachine(num_feature_maps=8, kernel_size=3, num_classes=10, std_dev=0.05)
    best_model.load_state_dict(torch.load("elm.pt", map_location=torch.device('cpu')))
    best_model.eval()
    create_montage(best_model, testloader, classes, save_path="result.png", num_images=36)
    print("Montage saved to result.png")
    # Return testloader, classes, and the best model instance for later testing/visualization.
    return testloader, classes, best_model_instance

if __name__ == "__main__":
    # Run training when executing train.py directly.
    train_models()
