import os
import sys

# Add the project root folder to sys.path so config can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import config
from utils.dataset import DefectDataset, get_transforms
from models.custom_cnn import CustomCNN
from models.resnet_model import get_resnet18_model
from models.efficientnet_model import get_efficientnet_model

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Runs a single training epoch through the data loader.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0   
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        # Reset gradients, forward pass, loss backprop, and optimizer step
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Accumulate metrics
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    """
    Evaluates the model performance on the validation split.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc

def train_model(model_name, device):
    print("=" * 60)
    print(f"Starting Training for Model: {model_name}")
    print("=" * 60)
    
    # 1. Load data splits
    train_dataset = DefectDataset(config.DATASET_SPLIT_DIR, "train", get_transforms("train"))
    val_dataset = DefectDataset(config.DATASET_SPLIT_DIR, "val", get_transforms("val"))
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    print(f"Loaded {len(train_dataset)} training images and {len(val_dataset)} validation images.")
    
    # 2. Instantiate Model and Optimizers
    if model_name == "custom_cnn":
        model = CustomCNN(num_classes=config.NUM_CLASSES)
        lr = config.LEARNING_RATE_CUSTOM
    elif model_name == "resnet18":
        model = get_resnet18_model(num_classes=config.NUM_CLASSES, fine_tune=True)
        lr = config.LEARNING_RATE_RESNET
    elif model_name == "efficientnet_b0":
        model = get_efficientnet_model(num_classes=config.NUM_CLASSES, fine_tune=True)
        lr = config.LEARNING_RATE_EFFICIENTNET
    else:
        print(f"Error: Unknown model {model_name}")
        return
        
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
    
    # 3. Setup history metrics tracking
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }
    
    best_val_acc = 0.0
    best_checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"best_{model_name}.pth")
    
    # 4. Training Loop (Fixed Epochs)
    start_time = time.time()
    
    for epoch in range(1, config.EPOCHS + 1):
        epoch_start = time.time()
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Save to history dictionary
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        
        duration = time.time() - epoch_start
        print(
            f"Epoch [{epoch}/{config.EPOCHS}] - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}% | "
            f"Time: {duration:.1f}s"
        )
        
        # Checkpoint Saving (Save based on Best Validation Accuracy)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f"--> Saved best model checkpoint (Val Acc: {val_acc*100:.2f}%, Val Loss: {val_loss:.4f})")
                
    total_time = time.time() - start_time
    print(f"Completed training {model_name} in {total_time/60:.2f} minutes.")
    print(f"Best Validation Accuracy achieved: {best_val_acc*100:.2f}%")
    
    # 5. Plot and save training curves using Matplotlib
    epochs_range = range(1, config.EPOCHS + 1)
    plt.figure(figsize=(12, 5))
    
    # Plot Loss Curves
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history["train_loss"], label="Train Loss", marker='o')
    plt.plot(epochs_range, history["val_loss"], label="Val Loss", marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} - Loss Curves")
    plt.legend()
    
    # Plot Accuracy Curves
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, [acc * 100 for acc in history["train_acc"]], label="Train Acc", marker='o')
    plt.plot(epochs_range, [acc * 100 for acc in history["val_acc"]], label="Val Acc", marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"{model_name} - Accuracy Curves")
    plt.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(config.OUTPUT_DIR, f"{model_name}_training_curves.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Training curves saved to: {plot_path}")
    
    return best_val_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train deep learning model for Quality Control defect detection.")
    parser.add_argument(
        "--model",
        type=str,
        choices=["custom_cnn", "resnet18", "efficientnet_b0", "all"],
        default="all",
        help="Select model to train: custom_cnn, resnet18, efficientnet_b0, or all."
    )
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device selected: {device}")
    
    models_to_train = ["custom_cnn", "resnet18", "efficientnet_b0"] if args.model == "all" else [args.model]
    
    for m_name in models_to_train:
        train_model(m_name, device)
