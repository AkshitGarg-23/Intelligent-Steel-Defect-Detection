import os
import sys

# Add the project root folder to sys.path so config and models can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from PIL import Image
import io

import config
from utils.dataset import DefectDataset, get_transforms
from models.custom_cnn import CustomCNN
from models.resnet_model import get_resnet18_model
from models.efficientnet_model import get_efficientnet_model

def evaluate_model(model_name, device):
    """
    Evaluates a specific model on the holdout test set.
    """
    print(f"Evaluating model: {model_name}...")
    
    # 1. Load data
    test_dataset = DefectDataset(config.DATASET_SPLIT_DIR, "test", get_transforms("test"))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    print(f"Loaded {len(test_dataset)} test images.")
    
    # 2. Instantiate and load Model weights
    if model_name == "custom_cnn":
        model = CustomCNN(num_classes=config.NUM_CLASSES)
    elif model_name == "resnet18":
        model = get_resnet18_model(num_classes=config.NUM_CLASSES, fine_tune=True)
    elif model_name == "efficientnet_b0":
        model = get_efficientnet_model(num_classes=config.NUM_CLASSES, fine_tune=True)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
        
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"best_{model_name}.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
        
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 3. Inference loop to measure Latency and predict
    y_true = []
    y_pred = []
    y_probs = []
    latencies = []
    misclassified = []
    
    softmax = nn.Softmax(dim=1)
    
    with torch.no_grad():
        for i, (image, label) in enumerate(test_loader):
            image, label = image.to(device), label.to(device)
            
            # Warm up latency by skipping first 5
            start_time = time.perf_counter()
            outputs = model(image)
            end_time = time.perf_counter()
            
            latencies.append((end_time - start_time) * 1000.0)
            
            probs = softmax(outputs)
            max_prob, predicted = torch.max(probs, 1)
            
            actual_idx = label.item()
            pred_idx = predicted.item()
            
            y_true.append(actual_idx)
            y_pred.append(pred_idx)
            y_probs.append(probs.cpu().numpy()[0])
            
            # Save misclassifications
            if actual_idx != pred_idx:
                img_path = test_dataset.image_paths[i]
                misclassified.append({
                    "image_path": img_path,
                    "actual": config.CLASSES[actual_idx],
                    "predicted": config.CLASSES[pred_idx],
                    "confidence": max_prob.item(),
                    "probabilities": probs.cpu().numpy()[0]
                })
                
    # 4. Calculate final metrics
    avg_latency = np.mean(latencies[5:])  # Ignore first 5 warmups
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=config.CLASSES, yticklabels=config.CLASSES
    )
    plt.title(f"Confusion Matrix - {model_name.upper()}", fontsize=14, fontweight='bold')
    plt.ylabel("Actual Defect", fontsize=12)
    plt.xlabel("Predicted Defect", fontsize=12)
    plt.xticks(rotation=15)
    plt.tight_layout()
    cm_path = os.path.join(config.OUTPUT_DIR, f"{model_name}_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    
    class_report = classification_report(y_true, y_pred, target_names=config.CLASSES)
    print(f"\nClassification Report for {model_name}:\n{class_report}")
    
    # Save Report details
    report_path = os.path.join(config.OUTPUT_DIR, f"{model_name}_classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Classification Report - {model_name}\n")
        f.write("="*50 + "\n")
        f.write(f"Total parameters: {total_params:,}\n")
        f.write(f"Trainable parameters: {trainable_params:,}\n")
        f.write(f"Avg Latency: {avg_latency:.2f} ms\n\n")
        f.write(class_report)
        
    metrics = {
        "model": model_name,
        "params": total_params,
        "trainable_params": trainable_params,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "latency_ms": avg_latency
    }
    
    return metrics, misclassified

def plot_misclassified(misclassified_all, save_path):
    """
    Plots a grid of misclassified images.
    """
    if len(misclassified_all) == 0:
        print("No misclassifications found to plot.")
        return
        
    num_samples = min(len(misclassified_all), 6)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    
    for i in range(len(axes)):
        if i < num_samples:
            sample = misclassified_all[i]
            img = Image.open(sample["image_path"])
            axes[i].imshow(img, cmap='gray' if img.mode == 'L' else None)
            axes[i].set_title(
                f"Actual: {sample['actual']}\nPred: {sample['predicted']} ({sample['confidence']*100:.1f}%)",
                fontsize=10, fontweight='bold', color='red'
            )
        axes[i].axis("off")
        
    plt.suptitle("Error Analysis: Misclassified Samples", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved misclassified samples visualization to: {save_path}")

def run_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation device: {device}")
    
    models = ["custom_cnn", "resnet18", "efficientnet_b0"]
    all_metrics = []
    all_misclassified = []
    
    for m_name in models:
        try:
            metrics, misclassified = evaluate_model(m_name, device)
            all_metrics.append(metrics)
            for m in misclassified:
                m["model"] = m_name
            all_misclassified.extend(misclassified)
        except Exception as e:
            print(f"Failed to evaluate {m_name}: {e}")
            
    if not all_metrics:
        print("Error: No models evaluated successfully.")
        return
        
    df = pd.DataFrame(all_metrics)
    
    df["accuracy"] = df["accuracy"] * 100
    df["precision"] = df["precision"] * 100
    df["recall"] = df["recall"] * 100
    df["f1_score"] = df["f1_score"] * 100
    
    df.columns = [
        "Model Architecture", "Total Params", "Trainable Params", 
        "Accuracy (%)", "Precision (%)", "Recall (%)", "F1 Score (%)", 
        "Latency per Image (ms)"
    ]
    
    comparison_str = df.to_string(index=False, formatters={
        "Total Params": lambda x: f"{x:,}",
        "Trainable Params": lambda x: f"{x:,}",
        "Accuracy (%)": lambda x: f"{x:.2f}%",
        "Precision (%)": lambda x: f"{x:.2f}%",
        "Recall (%)": lambda x: f"{x:.2f}%",
        "F1 Score (%)": lambda x: f"{x:.2f}%",
        "Latency per Image (ms)": lambda x: f"{x:.2f} ms"
    })
    
    print("\n" + "="*80 + "\nMODEL COMPARISON SUMMARY:\n" + "="*80 + f"\n{comparison_str}\n" + "="*80)
    
    comp_path = os.path.join(config.OUTPUT_DIR, "model_comparison.txt")
    with open(comp_path, "w") as f:
        f.write("="*80 + "\nMODEL COMPARISON SUMMARY\n" + "="*80 + f"\n{comparison_str}\n" + "="*80)
        
    for m_name in models:
        m_misclassified = [m for m in all_misclassified if m["model"] == m_name]
        if m_misclassified:
            plot_misclassified(m_misclassified, os.path.join(config.OUTPUT_DIR, f"misclassified_samples_{m_name}.png"))

if __name__ == "__main__":
    if sys.platform.startswith("win"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8') if hasattr(sys.stdout, 'buffer') else sys.stdout
    run_evaluation()
