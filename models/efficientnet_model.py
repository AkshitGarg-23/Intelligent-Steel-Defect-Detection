import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

def get_efficientnet_model(num_classes: int = 6, fine_tune: bool = True) -> nn.Module:
    """
    Loads pretrained EfficientNet-B0 and prepares it for transfer learning.
    """
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)
    
    # 1. Freeze all backbone layers first
    for param in model.parameters():
        param.requires_grad = False
        
    # 2. Replace the classification head
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    # 3. Unfreeze last feature blocks (7 and 8) for fine-tuning
    if fine_tune:
        for param in model.features[7].parameters():
            param.requires_grad = True
        for param in model.features[8].parameters():
            param.requires_grad = True
        for param in model.classifier.parameters():
            param.requires_grad = True
            
    return model

if __name__ == "__main__":
    from torchinfo import summary
    model = get_efficientnet_model(num_classes=6, fine_tune=True)
    print("EfficientNet-B0 Architecture Summary:")
    summary(model, input_size=(1, 3, 224, 224), col_names=["input_size", "output_size", "num_params", "trainable"])
