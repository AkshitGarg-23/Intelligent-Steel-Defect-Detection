import os
import sys

# Add the project root folder to sys.path so config and models can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io
import time
import requests
import numpy as np
import pandas as pd
import torch
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt

import config
from models.custom_cnn import CustomCNN
from models.resnet_model import get_resnet18_model
from models.efficientnet_model import get_efficientnet_model
from utils.dataset import get_transforms
from utils.gradcam import GradCAM, overlay_heatmap_on_image

# Load API URL from Environment Variable, fallback to local URL
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Page configuration
st.set_page_config(
    page_title="Industrial Defect Inspection System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Caching models for local Grad-CAM visualization
@st.cache_resource
def load_local_model(model_name: str):
    device = torch.device("cpu")
    if model_name == "custom_cnn":
        model = CustomCNN(num_classes=config.NUM_CLASSES)
    elif model_name == "resnet18":
        model = get_resnet18_model(num_classes=config.NUM_CLASSES, fine_tune=True)
    elif model_name == "efficientnet_b0":
        model = get_efficientnet_model(num_classes=config.NUM_CLASSES, fine_tune=True)
    else:
        raise ValueError(f"Unknown model choice: {model_name}")
        
    path = os.path.join(config.CHECKPOINT_DIR, f"best_{model_name}.pth")
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

# Initialize session state for prediction history
if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar navigation
st.sidebar.title("🏭 QC System")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["Home", "Dataset Analysis", "Predict", "Model Comparison", "About"]
)

# ==========================================
# PAGE 1: HOME
# ==========================================
if page == "Home":
    st.title("🏭 Industrial Defect Quality Control Dashboard")
    st.markdown("### AI-Powered Real-Time Quality Inspection")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        #### Problem Statement
        In modern steel sheet manufacturing, rolling machinery runs at high speeds. Defects such as Crazing, Inclusion, Patches, Pitted Surfaces, Rolled-in Scale, and Scratches occur due to temperature changes or mechanical impurities. Manual inspection is slow and error-prone.
        
        #### Project Goals & Solution
        - **Real-Time Classification**: Automatically identify defect classes using Deep Learning.
        - **Visual Interpretability**: Highlight defect locations using Grad-CAM heatmaps.
        - **Architectural Comparisons**: Compare Custom CNN (scratch baseline), ResNet18 (transfer learning), and EfficientNet-B0 (transfer learning) to evaluate latency-accuracy trade-offs.
        """)
    
    with col2:
        st.subheader("System Pipeline")
        st.code("""
[Steel Sheet Line]
       │ (Camera Input)
       ▼
[FastAPI Predict Backend]
  ├── Transform & Normalize
  ├── Custom CNN
  ├── ResNet18
  └── EfficientNet-B0
       │
       ▼
[QC Streamlit GUI]
  ├── Classification
  └── Grad-CAM Explainability
        """)

# ==========================================
# PAGE 2: DATASET ANALYSIS
# ==========================================
elif page == "Dataset Analysis":
    st.title("📊 Dataset Exploratory Analysis")
    
    col1, col2 = st.columns([1, 1])
    
    dist_path = os.path.join(config.OUTPUT_DIR, "class_distribution.png")
    sample_path = os.path.join(config.OUTPUT_DIR, "sample_images.png")
    pixel_path = os.path.join(config.OUTPUT_DIR, "pixel_distribution.png")
    
    with col1:
        st.subheader("Class Distribution across Splits")
        if os.path.exists(dist_path):
            st.image(dist_path, use_container_width=True)
        else:
            st.warning("Distribution chart not found. Please run the EDA script first.")
            
        st.subheader("Defect Pixel Intensity Distribution")
        if os.path.exists(pixel_path):
            st.image(pixel_path, use_container_width=True)
        else:
            st.warning("Pixel distribution chart not found. Please run the EDA script first.")
            
    with col2:
        st.subheader("Sample Defect Visualizations")
        if os.path.exists(sample_path):
            st.image(sample_path, use_container_width=True)
        else:
            st.warning("Sample image grid not found. Please run the EDA script first.")

# ==========================================
# PAGE 3: PREDICT
# ==========================================
elif page == "Predict":
    st.title("🔍 Real-time Surface Defect Prediction")
    
    col_input, col_pred = st.columns([1, 2])
    
    example_base_dir = os.path.join(config.DATASET_SPLIT_DIR, "test")
    uploaded_image = None
    img_path_for_cam = None
    
    with col_input:
        st.subheader("Inspection Settings")
        model_choice = st.selectbox("Select Classifier Model", ["resnet18", "custom_cnn", "efficientnet_b0"])
        source = st.radio("Image Source", ["Upload Image", "Use Dataset Example"])
        
        if source == "Upload Image":
            uploaded_file = st.file_uploader("Upload Image File", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                uploaded_image = Image.open(uploaded_file).convert("RGB")
                temp_path = os.path.join(config.OUTPUT_DIR, "temp_uploaded.jpg")
                uploaded_image.save(temp_path)
                img_path_for_cam = temp_path
        else:
            if os.path.exists(example_base_dir):
                cls_choice = st.selectbox("Select Defect Category", config.CLASSES)
                test_class_dir = os.path.join(example_base_dir, cls_choice)
                example_files = [f for f in os.listdir(test_class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                selected_fn = st.selectbox("Select File", example_files)
                img_path_for_cam = os.path.join(test_class_dir, selected_fn)
                uploaded_image = Image.open(img_path_for_cam).convert("RGB")
            else:
                st.warning("Split dataset files not found. Run data_split.py first.")
                
        run_cam = st.checkbox("Show Grad-CAM Heatmap", value=True)
        
    with col_pred:
        st.subheader("Prediction Results")
        if uploaded_image is not None:
            col_img1, col_img2 = st.columns([1, 1])
            
            with col_img1:
                st.image(uploaded_image, caption="Original Sample", use_container_width=True)
                
            try:
                # Convert image to byte stream
                img_byte_arr = io.BytesIO()
                uploaded_image.save(img_byte_arr, format='JPEG')
                img_bytes = img_byte_arr.getvalue()
                
                # Post to FastAPI backend
                files = {"file": ("image.jpg", img_bytes, "image/jpeg")}
                response = requests.post(f"{API_URL}/predict?model_name={model_choice}", files=files)
                
                if response.status_code == 200:
                    res = response.json()
                    pred_class = res["predicted_class"]
                    confidence = res["confidence"]
                    probs = res["probabilities"]
                    lat_ms = res["inference_time_ms"]
                    
                    # Store in session state history
                    st.session_state.history.append({
                        "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "Model": model_choice,
                        "Prediction": pred_class,
                        "Confidence": f"{confidence * 100:.2f}%",
                        "Latency": f"{lat_ms:.2f} ms"
                    })
                    
                    st.success(f"**Predicted Class**: {pred_class.replace('_', ' ').upper()}")
                    
                    m1, m2 = st.columns([1, 1])
                    m1.metric("Confidence", f"{confidence * 100:.2f}%")
                    m2.metric("Inference Latency", f"{lat_ms:.2f} ms")
                    
                    # Probabilities horizontal bar chart
                    df_probs = pd.DataFrame(list(probs.items()), columns=["Defect", "Probability"])
                    df_probs = df_probs.sort_values(by="Probability", ascending=True)
                    
                    fig, ax = plt.subplots(figsize=(6, 3))
                    bars = ax.barh(df_probs["Defect"], df_probs["Probability"], color="#1f77b4")
                    ax.set_xlim(0, 1.05)
                    for bar in bars:
                        width = bar.get_width()
                        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width*100:.1f}%', 
                                va='center', ha='left', fontsize=8, fontweight='bold')
                    plt.title("Class Probability Chart", fontsize=10, fontweight='bold')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
                    # Generate Grad-CAM heatmap locally
                    if run_cam and img_path_for_cam is not None:
                        with col_img2:
                            with st.spinner("Calculating activation maps..."):
                                local_model = load_local_model(model_choice)
                                if model_choice == "custom_cnn":
                                    target_layer = local_model.conv4
                                elif model_choice == "resnet18":
                                    target_layer = local_model.layer4[-1]
                                elif model_choice == "efficientnet_b0":
                                    target_layer = local_model.features[8]
                                    
                                cam = GradCAM(local_model, target_layer)
                                val_transform = get_transforms("val")
                                img_tensor = val_transform(uploaded_image)
                                
                                pred_idx = config.CLASSES.index(pred_class)
                                heatmap = cam.generate_heatmap(img_tensor, target_class=pred_idx)
                                cam_overlay = overlay_heatmap_on_image(img_path_for_cam, heatmap, alpha=0.45)
                                st.image(cam_overlay, caption=f"Grad-CAM Attention Map: {pred_class}", use_container_width=True)
                                cam.remove_hooks()
                else:
                    st.error(f"Error from FastAPI backend: {response.status_code}")
            except Exception as e:
                st.warning(f"FastAPI backend offline. Please verify that the backend is running at {API_URL}.")
                st.error(f"Connection Error: {e}")
        else:
            st.info("Upload an image file or choose an example from the sidebar dataset folders.")
            
    # Session History table
    st.markdown("---")
    st.subheader("Session Inspection Log")
    if st.session_state.history:
        st.dataframe(pd.DataFrame(st.session_state.history).iloc[::-1], use_container_width=True)
    else:
        st.write("No inspections recorded yet.")

# ==========================================
# PAGE 4: MODEL COMPARISON
# ==========================================
elif page == "Model Comparison":
    st.title("📈 Model Comparison & Benchmark Benchmarks")
    
    st.subheader("Evaluation holdout test statistics:")
    comp_path = os.path.join(config.OUTPUT_DIR, "model_comparison.txt")
    if os.path.exists(comp_path):
        with open(comp_path, "r") as f:
            st.code(f.read())
    else:
        st.warning("Please run evaluate.py to populate actual test benchmarks.")

# ==========================================
# PAGE 5: ABOUT
# ==========================================
elif page == "About":
    st.title("ℹ️ About the Project")
    st.markdown("""
    This project is an end-to-end computer vision and deep learning system built to automate steel defect inspection.
    
    #### Key Features Implemented:
    - **Custom CNN**: Implemented from scratch using PyTorch layers.
    - **ResNet18 / EfficientNet-B0 Fine-Tuning**: Transfer learning unfreezing final network layers.
    - **FastAPI**: Synchronous endpoint for model inference.
    - **Streamlit**: Web Dashboard containing EDA and visual defect inspection.
    - **Grad-CAM**: Gradient-weighted explainability displaying model activation heatmaps.
    
    *Developed for AI/ML engineering internships.*
    """)
