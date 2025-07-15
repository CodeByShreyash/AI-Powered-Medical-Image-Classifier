import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0, resnet18
from PIL import Image
import numpy as np
import os
from utils.gradcam import GradCAM, overlay_heatmap

# --- CLASS LABELS ---
efficientnet_classes = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
resnet18_classes = ['benign', 'malignant', 'suspicious', 'other']

# --- MODEL LOADING FUNCTIONS ---
@st.cache_resource
def load_efficientnet():
    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(efficientnet_classes))
    model_path = os.path.join(os.path.dirname(__file__), "model", "model.pth")
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

@st.cache_resource
def load_resnet18():
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(resnet18_classes))
    model_path = os.path.join(os.path.dirname(__file__), "model", "resnet18_skin_cancer.pth")
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

# --- IMAGE TRANSFORM ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- PAGE CONFIG ---
st.set_page_config(page_title="SkinSight AI", layout="wide")

# --- HERO SECTION ---
with st.container():
    st.markdown("<h1 style='text-align: center;'>Instant Skin Lesion<br><span style='color:#00d4aa'>Diagnosis with AI</span></h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Upload a photo. Get prediction & visual explanation in seconds.</p>", unsafe_allow_html=True)

st.markdown("---")

# --- MAIN SECTION ---
st.subheader("AI-Powered Skin Analysis")
st.caption("Upload a dermoscopic image to get instant analysis with visual explanations")

col_left, col_right = st.columns(2)

# --- LEFT: Upload & Predict ---
with col_left:
    uploaded_file = st.file_uploader("Upload Skin Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        model_option = st.selectbox("Select Model", ["EfficientNet B0 (7 classes)", "ResNet18 (4 classes)"], index=0)

        # Apply transformation
        input_tensor = transform(image).unsqueeze(0)

        # Model selection
        if "EfficientNet" in model_option:
            model = load_efficientnet()
            class_names = efficientnet_classes
            target_layer = model.features[-1]
        else:
            model = load_resnet18()
            class_names = resnet18_classes
            target_layer = model.layer4[-1]

        # Grad-CAM setup
        gradcam = GradCAM(model, target_layer)
        heatmap = gradcam.generate(input_tensor)

        # Prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1).squeeze().numpy()
            pred_idx = np.argmax(probs)
            pred_class = class_names[pred_idx]
            confidence = probs[pred_idx]

# --- RIGHT: Results ---
with col_right:
    if uploaded_file:
        st.markdown("### 🏷 Prediction Result")
        st.success(f"**{pred_class.upper()}** — {confidence * 100:.2f}% Confidence")

        st.markdown("### 📊 Class Probabilities")
        for i, class_name in enumerate(class_names):
            percent = float(probs[i]) * 100
            st.markdown(f"**{class_name.upper()}** — {percent:.2f}%")
            st.progress(float(probs[i]))

        st.markdown("### 🔍 Visual Explanation (Grad-CAM)")
        image_np = np.array(image.resize((224, 224)))
        cam_overlay = overlay_heatmap(heatmap, image_np)
        st.image(cam_overlay, use_container_width=True)

st.markdown("---")

# --- INFO ---
st.subheader("Model & Dataset Information")
st.caption("Our AI models are trained on validated medical datasets and thoroughly tested for accuracy.")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("*Model Architectures*")
    st.write("EfficientNet B0")
    st.write("ResNet18")
    st.write("Input Size: 224x224")

with col2:
    st.markdown("*Training Dataset*")
    st.write("ISIC 2018")
    st.write("Images: 10,015")
    st.write("EfficientNet: 7 classes")
    st.write("ResNet18: 4 classes")

with col3:
    st.markdown("*Performance (Example)*")
    st.write("Accuracy: ~88%")
    st.write("Precision: ~89%")
    st.write("Recall: ~90%")

# --- DISCLAIMER ---
st.warning("""
⚠ *Important Disclaimer*  
This is a research prototype and educational tool. The predictions made by this AI system are not intended for medical diagnosis or treatment. Please consult certified professionals for clinical decisions.
""")

# --- FOOTER ---
st.markdown("---")
footer1, footer2 = st.columns([1, 3])
with footer1:
    st.markdown("### SkinSight AI")
    st.caption("AI-powered diagnosis for skin lesions.")
with footer2:
    st.markdown("""
- [About](#)
- [Model Info](#)
- [Disclaimer](#)

📧 contact@skinsight.ai  
Made using Streamlit & PyTorch  
    """)


# import streamlit as st
# import torch
# import torch.nn as nn
# from torchvision import transforms
# from torchvision.models import efficientnet_b0, resnet18
# from PIL import Image
# import numpy as np
# from utils.gradcam import GradCAM, overlay_heatmap

# # --- CLASS LABELS ---
# efficientnet_classes = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
# resnet18_classes = ['benign', 'malignant', 'suspicious', 'other']

# # --- MODEL LOADING FUNCTIONS ---
# @st.cache_resource
# def load_efficientnet():
#     model = efficientnet_b0(weights=None)
#     model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(efficientnet_classes))
#     model.load_state_dict(torch.load("model/model.pth", map_location="cpu"))
#     model.eval()
#     return model

# @st.cache_resource
# def load_resnet18():
#     model = resnet18(weights=None)
#     model.fc = nn.Linear(model.fc.in_features, len(resnet18_classes))
#     model.load_state_dict(torch.load("model/resnet18_skin_cancer.pth", map_location="cpu"))
#     model.eval()
#     return model

# # --- IMAGE TRANSFORM ---
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
# ])

# # --- PAGE CONFIG ---
# st.set_page_config(page_title="SkinSight AI", layout="wide")

# # --- HERO SECTION ---
# with st.container():
#     st.markdown("<h1 style='text-align: center;'>Instant Skin Lesion<br><span style='color:#00d4aa'>Diagnosis with AI</span></h1>", unsafe_allow_html=True)
#     st.markdown("<p style='text-align: center;'>Upload a photo. Get prediction & visual explanation in seconds.</p>", unsafe_allow_html=True)
#     col1, col2, col3 = st.columns([1, 2, 1])
#     # with col2:
#     #     st.button("Get Started")
#     #     st.button("Learn More")

# st.markdown("---")

# # --- MAIN SECTION ---
# st.subheader("AI-Powered Skin Analysis")
# st.caption("Upload a dermoscopic image to get instant analysis with visual explanations")

# col_left, col_right = st.columns(2)

# # --- LEFT: Upload & Predict ---
# with col_left:
#     uploaded_file = st.file_uploader("Upload Skin Image", type=["jpg", "jpeg", "png"])

#     if uploaded_file:
#         image = Image.open(uploaded_file).convert("RGB")
#         st.image(image, caption="Uploaded Image", use_container_width=True)

#         model_option = st.selectbox("Select Model", ["EfficientNet B0 (7 classes)", "ResNet18 (4 classes)"], index=0)

#         # Apply transformation
#         input_tensor = transform(image).unsqueeze(0)

#         # Model selection
#         if "EfficientNet" in model_option:
#             model = load_efficientnet()
#             class_names = efficientnet_classes
#             target_layer = model.features[-1]
#         else:
#             model = load_resnet18()
#             class_names = resnet18_classes
#             target_layer = model.layer4[-1]

#         # Grad-CAM setup
#         gradcam = GradCAM(model, target_layer)
#         heatmap = gradcam.generate(input_tensor)

#         # Prediction
#         with torch.no_grad():
#             outputs = model(input_tensor)
#             probs = torch.softmax(outputs, dim=1).squeeze().numpy()
#             pred_idx = np.argmax(probs)
#             pred_class = class_names[pred_idx]
#             confidence = probs[pred_idx]

# # --- RIGHT: Results ---
# with col_right:
#     if uploaded_file:
#         st.markdown("### 🏷 Prediction Result")
#         st.success(f"**{pred_class.upper()}** — {confidence * 100:.2f}% Confidence")

#         st.markdown("### 📊 Class Probabilities")
#         for i, class_name in enumerate(class_names):
#             percent = float(probs[i]) * 100
#             st.markdown(f"**{class_name.upper()}** — {percent:.2f}%")
#             st.progress(float(probs[i]))  # ✅ FIX: Convert float32 to native float

#         st.markdown("### 🔍 Visual Explanation (Grad-CAM)")
#         image_np = np.array(image.resize((224, 224)))
#         cam_overlay = overlay_heatmap(heatmap, image_np)
#         st.image(cam_overlay, use_container_width=True)  # ✅ FIXED DEPRECATION

# st.markdown("---")

# # --- INFO ---
# st.subheader("Model & Dataset Information")
# st.caption("Our AI models are trained on validated medical datasets and thoroughly tested for accuracy.")

# col1, col2, col3 = st.columns(3)
# with col1:
#     st.markdown("*Model Architectures*")
#     st.write("EfficientNet B0")
#     st.write("ResNet18")
#     st.write("Input Size: 224x224")

# with col2:
#     st.markdown("*Training Dataset*")
#     st.write("ISIC 2018")
#     st.write("Images: 10,015")
#     st.write("EfficientNet: 7 classes")
#     st.write("ResNet18: 4 classes")

# with col3:
#     st.markdown("*Performance (Example)*")
#     st.write("Accuracy: ~88%")
#     st.write("Precision: ~89%")
#     st.write("Recall: ~90%")

# # --- DISCLAIMER ---
# st.warning("""
# ⚠ *Important Disclaimer*  
# This is a research prototype and educational tool. The predictions made by this AI system are not intended for medical diagnosis or treatment. Please consult certified professionals for clinical decisions.
# """)

# # --- FOOTER ---
# st.markdown("---")
# footer1, footer2 = st.columns([1, 3])
# with footer1:
#     st.markdown("### SkinSight AI")
#     st.caption("AI-powered diagnosis for skin lesions.")
# with footer2:
#     st.markdown("""
# - [About](#)
# - [Model Info](#)
# - [Disclaimer](#)

# 📧 contact@skinsight.ai  
# Made using Streamlit & PyTorch  
#     """)
