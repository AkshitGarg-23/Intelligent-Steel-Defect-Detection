import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torchinfo import summary

def get_resnet18_model(num_classes: int = 6, fine_tune: bool = True) -> nn.Module:
    """
    Loads pretrained ResNet18 and prepares it for transfer learning/fine-tuning.
    
    Args:
        num_classes: Number of classification categories.
        fine_tune: If True, unfreezes layer4 (the last residual block) for fine-tuning.
                   If False, freezes the entire backbone and only trains the final FC layer.
    """
    # Load pretrained ResNet18 with default ImageNet weights
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    
    # 1. Freeze all backbone layers first
    for param in model.parameters():
        param.requires_grad = False
        
    # 2. Replace the classification head
    # A new layer instantiated in PyTorch has requires_grad = True by default
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    # 3. Optionally unfreeze the last residual block (layer4) for fine-tuning
    if fine_tune:
        for param in model.layer4.parameters():
            param.requires_grad = True
            
    return model

if __name__ == "__main__":
    import sys
    import io
    if sys.platform.startswith("win"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        
    # Create the model in fine-tune mode
    model = get_resnet18_model(num_classes=6, fine_tune=True)
    
    print("ResNet18 Model Architecture Summary (Fine-Tuning Last Residual Block):")
    # Using torchinfo.summary to show trainable vs frozen parameters
    summary(model, input_size=(1, 3, 224, 224), col_names=["input_size", "output_size", "num_params", "trainable"])
