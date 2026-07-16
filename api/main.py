import os
import sys

# Add the project root folder to sys.path so config, models, and utils can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io
import time
import torch
import torch.nn as nn
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse

import config
from models.custom_cnn import CustomCNN
from models.resnet_model import get_resnet18_model
from models.efficientnet_model import get_efficientnet_model
from utils.dataset import get_transforms

app = FastAPI(
    title="Industrial Quality Control API",
    description="Sync FastAPI server for hot-rolled steel surface defect classification."
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models = {}
transform = None

@app.on_event("startup")
def load_models():
    global models, transform
    print(f"Loading models on device: {device}")
    
    transform = get_transforms("val")
    
    # 1. Load Custom CNN
    try:
        cnn = CustomCNN(num_classes=config.NUM_CLASSES)
        path = os.path.join(config.CHECKPOINT_DIR, "best_custom_cnn.pth")
        if os.path.exists(path):
            cnn.load_state_dict(torch.load(path, map_location=device))
            print("Loaded Custom CNN weights.")
            cnn.to(device).eval()
            models["custom_cnn"] = cnn
        else:
            print(f"Warning: Checkpoint not found at {path}. Custom CNN will not be loaded.")
    except Exception as e:
        print(f"Failed to load Custom CNN: {e}")
        
    # 2. Load ResNet18
    try:
        resnet = get_resnet18_model(num_classes=config.NUM_CLASSES, fine_tune=True)
        path = os.path.join(config.CHECKPOINT_DIR, "best_resnet18.pth")
        if os.path.exists(path):
            resnet.load_state_dict(torch.load(path, map_location=device))
            print("Loaded ResNet18 weights.")
            resnet.to(device).eval()
            models["resnet18"] = resnet
        else:
            print(f"Warning: Checkpoint not found at {path}. ResNet18 will not be loaded.")
    except Exception as e:
        print(f"Failed to load ResNet18: {e}")

    # 3. Load EfficientNet-B0
    try:
        effnet = get_efficientnet_model(num_classes=config.NUM_CLASSES, fine_tune=True)
        path = os.path.join(config.CHECKPOINT_DIR, "best_efficientnet_b0.pth")
        if os.path.exists(path):
            effnet.load_state_dict(torch.load(path, map_location=device))
            print("Loaded EfficientNet-B0 weights.")
            effnet.to(device).eval()
            models["efficientnet_b0"] = effnet
        else:
            print(f"Warning: Checkpoint not found at {path}. EfficientNet-B0 will not be loaded.")
    except Exception as e:
        print(f"Failed to load EfficientNet-B0: {e}")

@app.get("/")
def read_root():
    return {
        "status": "Online",
        "project": "Industrial Defect Quality Control",
        "available_models": list(models.keys())
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "device": str(device)}

@app.post("/predict")
def predict(
    file: UploadFile = File(...),
    model_name: str = Query("resnet18", enum=["custom_cnn", "resnet18", "efficientnet_b0"])
):
    if model_name not in models:
        return JSONResponse(
            status_code=400,
            content={"error": f"Model '{model_name}' is not loaded."}
        )
        
    try:
        img_bytes = file.file.read()
        if len(img_bytes) == 0:
            return JSONResponse(
                status_code=400,
                content={"error": "Uploaded image file is empty."}
            )
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": f"Invalid image file: {e}"}
        )
        
    start_time = time.perf_counter()
    
    # Preprocess image and add batch dimension
    img_tensor = transform(image).unsqueeze(0).to(device)
    selected_model = models[model_name]
    
    with torch.no_grad():
        outputs = selected_model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)
        
    duration_ms = (time.perf_counter() - start_time) * 1000.0
    
    pred_idx = predicted.item()
    pred_class = config.CLASSES[pred_idx]
    probs_dict = {config.CLASSES[i]: float(probs[0][i].item()) for i in range(config.NUM_CLASSES)}
    
    return {
        "predicted_class": pred_class,
        "confidence": float(confidence.item()),
        "probabilities": probs_dict,
        "inference_time_ms": float(duration_ms)
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
