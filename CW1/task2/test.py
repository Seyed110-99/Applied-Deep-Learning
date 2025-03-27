"""
GenAI usage statement:
This file was generated with assistive help from a generative AI tool to comment the code and suggest optimised methods for faster performance.
The original code was written by the user, and the AI tool provided suggestions for comments and structure.
"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from task import (MyExtremeLearningMachine, MyEnsembleELM, evaluate_model)


def print_summary(title, metrics_dict):
    """Prints a summary of the performance metrics for the models.
    This function was made with the help of Generative AI.
    """
    print(f"\n=== {title} Performance Summary ===")
    print("Epoch/Checkpoint\tAccuracy (%)\tMacro F1 (%)")
    for key in sorted(metrics_dict.keys(), key=lambda x: (x != "Full", x)):
        acc, f1 = metrics_dict[key]
        print(f"{key:16s}\t{acc:6.2f}\t\t{f1:6.2f}")
    print("========================================\n")

def test_models():
    # -------------------------
    # Data Preparation for Testing
    # -------------------------
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    batch_size = 32
    test_dataset = datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    # -------------------------
    # Evaluate Plain ELM and its Checkpoints
    # -------------------------
    print(f"\nComparing Plain ELM with checkpoints saved")
    plain_metrics = {}
    Normal_elm = MyExtremeLearningMachine(num_feature_maps=8, kernel_size=3, num_classes=10, std_dev=0.05)
    
    # Full model
    Normal_elm.load_state_dict(torch.load("elm.pt", map_location=torch.device('cpu'), weights_only=True))
    Normal_elm.eval()
    plain_metrics["Full"] = evaluate_model(Normal_elm, testloader)
    
    # Checkpoint epoch 1
    Normal_elm.load_state_dict(torch.load("checkpoint_ELM_Epoch1.pt", map_location=torch.device('cpu'), weights_only=True))
    Normal_elm.eval()
    plain_metrics["Epoch 1"] = evaluate_model(Normal_elm, testloader)
    
    # Checkpoint epoch 3
    Normal_elm.load_state_dict(torch.load("checkpoint_ELM_Epoch3.pt", map_location=torch.device('cpu'), weights_only=True))
    Normal_elm.eval()
    plain_metrics["Epoch 3"] = evaluate_model(Normal_elm, testloader)
    
    print_summary("Plain ELM", plain_metrics)

    # -------------------------
    # Evaluate ELM with MixUp and its Checkpoints
    # -------------------------
    print(f"\nComparing ELM with MixUp with checkpoints saved")
    mixup_metrics = {}
    Mixup_elm = MyExtremeLearningMachine(num_feature_maps=8, kernel_size=3, num_classes=10, std_dev=0.05)
    
    # Full model
    Mixup_elm.load_state_dict(torch.load("elm_mixup.pt", map_location=torch.device('cpu'), weights_only=True))
    Mixup_elm.eval()
    mixup_metrics["Full"] = evaluate_model(Mixup_elm, testloader)
    
    # Checkpoint epoch 1
    Mixup_elm.load_state_dict(torch.load("checkpoint_ELM_Mixup_Epoch1.pt", map_location=torch.device('cpu'), weights_only=True))
    Mixup_elm.eval()
    mixup_metrics["Epoch 1"] = evaluate_model(Mixup_elm, testloader)
    
    # Checkpoint epoch 3
    Mixup_elm.load_state_dict(torch.load("checkpoint_ELM_Mixup_Epoch3.pt", map_location=torch.device('cpu'), weights_only=True))
    Mixup_elm.eval()
    mixup_metrics["Epoch 3"] = evaluate_model(Mixup_elm, testloader)
    
    print_summary("ELM with MixUp", mixup_metrics)

    # -------------------------
    # Evaluate Ensemble ELM Models and their Checkpoints
    # -------------------------
    print(f"\nComparing Ensemble ELM with checkpoints saved")
    ensemble_metrics = {}
    # For the full ensemble, ensure you instantiate with the same number as was saved.
    Ensemble_elm = MyEnsembleELM(num_models=5, num_feature_maps=8, kernel_size=3, 
                                 num_classes=10, std_dev=0.05, seed=42)
    Ensemble_elm.load_state_dict(torch.load("ensemble_elm.pt", map_location=torch.device('cpu'), weights_only=True))
    Ensemble_elm.eval()
    ensemble_metrics["Full"] = evaluate_model(Ensemble_elm, testloader)
    
    # Load checkpoint with 1 model
    checkpoint = torch.load("ensemble_elm_check1.pt", map_location=torch.device('cpu'))
    num_models_checkpoint = checkpoint["num_models"]
    Ensemble_elm_check_1 = MyEnsembleELM(num_models=num_models_checkpoint, num_feature_maps=8,
                                         kernel_size=3, num_classes=10, std_dev=0.05, seed=42)
    for i in range(num_models_checkpoint):
        Ensemble_elm_check_1.models[i].load_state_dict(checkpoint["ensemble_state"][i])
    Ensemble_elm_check_1.eval()
    ensemble_metrics[f"{num_models_checkpoint} model"] = evaluate_model(Ensemble_elm_check_1, testloader)
    
    # Load checkpoint with 3 models
    checkpoint = torch.load("ensemble_elm_check3.pt", map_location=torch.device('cpu'))
    num_models_checkpoint = checkpoint["num_models"]
    Ensemble_elm_check_3 = MyEnsembleELM(num_models=num_models_checkpoint, num_feature_maps=8,
                                         kernel_size=3, num_classes=10, std_dev=0.05, seed=42)
    for i in range(num_models_checkpoint):
        Ensemble_elm_check_3.models[i].load_state_dict(checkpoint["ensemble_state"][i])
    Ensemble_elm_check_3.eval()
    ensemble_metrics[f"{num_models_checkpoint} models"] = evaluate_model(Ensemble_elm_check_3, testloader)
    
    print_summary("Ensemble ELM", ensemble_metrics)

    # -------------------------
    # Evaluate Ensemble ELM with MixUp and their Checkpoints
    # -------------------------
    print(f"\nComparing Ensemble ELM with MixUp with checkpoints saved")
    ensemble_mixup_metrics = {}
    Ensemble_elm_mixup = MyEnsembleELM(num_models=5, num_feature_maps=8, kernel_size=3,
                                       num_classes=10, std_dev=0.05, seed=42)
    Ensemble_elm_mixup.load_state_dict(torch.load("ensemble_elm_mixup.pt", map_location=torch.device('cpu'), weights_only=True))
    Ensemble_elm_mixup.eval()
    ensemble_mixup_metrics["Full"] = evaluate_model(Ensemble_elm_mixup, testloader)
    
    checkpoint = torch.load("ensemble_elm_mixup_check1.pt", map_location=torch.device('cpu'))
    num_models_checkpoint = checkpoint["num_models"]
    Ensemble_elm_mixup_check_1 = MyEnsembleELM(num_models=num_models_checkpoint, num_feature_maps=8,
                                               kernel_size=3, num_classes=10, std_dev=0.05, seed=42)
    for i in range(num_models_checkpoint):
        Ensemble_elm_mixup_check_1.models[i].load_state_dict(checkpoint["ensemble_state"][i])
    Ensemble_elm_mixup_check_1.eval()
    ensemble_mixup_metrics[f"{num_models_checkpoint} model"] = evaluate_model(Ensemble_elm_mixup_check_1, testloader)
    
    checkpoint = torch.load("ensemble_elm_mixup_check3.pt", map_location=torch.device('cpu'))
    num_models_checkpoint = checkpoint["num_models"]
    Ensemble_elm_mixup_check_3 = MyEnsembleELM(num_models=num_models_checkpoint, num_feature_maps=8,
                                               kernel_size=3, num_classes=10, std_dev=0.05, seed=42)
    for i in range(num_models_checkpoint):
        Ensemble_elm_mixup_check_3.models[i].load_state_dict(checkpoint["ensemble_state"][i])
    Ensemble_elm_mixup_check_3.eval()
    ensemble_mixup_metrics[f"{num_models_checkpoint} models"] = evaluate_model(Ensemble_elm_mixup_check_3, testloader)
    
    print_summary("Ensemble ELM with MixUp", ensemble_mixup_metrics)

if __name__ == "__main__":
    test_models()
