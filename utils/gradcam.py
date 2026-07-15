import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.cm as cm
from PIL import Image

class GradCAM:
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping) implementation.
    Generates class activation heatmaps to explain model predictions.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register forward and backward hooks to capture activations and gradients
        self.forward_hook = self.target_layer.register_forward_hook(self._save_activations)
        self.backward_hook = self.target_layer.register_full_backward_hook(self._save_gradients)
        
    def _save_activations(self, module, input, output):
        self.activations = output.detach()
        
    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
        
    def generate_heatmap(self, input_tensor, target_class=None):
        """
        Generates a 2D Grad-CAM heatmap normalized to [0, 1].
        """
        self.model.eval()
        
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
            
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()
            
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1.0
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Compute weights as mean gradient per channel
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # Compute weighted sum of activations
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        
        # Apply ReLU to keep only positive contributions
        cam = F.relu(cam)
        
        # Resize activation map to match input resolution
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        
        # Normalize between 0 and 1
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
            
        return cam
        
    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()

def overlay_heatmap_on_image(img_path, heatmap, alpha=0.4):
    """
    Overlays the 2D heatmap on top of the original inspection image.
    """
    orig_img = Image.open(img_path).convert("RGB")
    orig_img_resized = orig_img.resize((224, 224))
    
    colormap = cm.get_cmap("jet")
    heatmap_colored = colormap(heatmap)
    
    # Keep RGB channels and convert to uint8 range (0-255)
    heatmap_colored = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)
    heatmap_img = Image.fromarray(heatmap_colored)
    
    # Blend the original image and heatmap together
    blended_img = Image.blend(orig_img_resized, heatmap_img, alpha=alpha)
    return blended_img
