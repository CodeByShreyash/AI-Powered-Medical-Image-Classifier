# app.py

import streamlit as st
import torch
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from utils.predict import predict
from utils.gradcam import GradCAM, overlay_heatmap
from torchvision.models import efficientnet_b0
import os

# --- Page Configuration ---
st.set_page_config(page_title="SkinSight AI", layout="centered")

# --- Custom CSS (optional for cleaner UI) ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .css-1d391kg { padding-top: 1rem; }
    </style>
""", unsafe_allow_html=True)

# --- Logo and Title ---
if os.path.exists("assets/logo.png"):
    st.image("assets/logo.png", width=150)
st.markdown("## :green[Instant Skin Lesion Diagnosis with AI]")
st.markdown("Upload a photo. Get prediction & visual explanation in seconds.")

col1, col2 = st.columns([1, 1])
with col1:
    st.button("🚀 Get Started")
with col2:
    st.button("ℹ️ Learn More")

st.markdown("---")

# --- Upload Image ---
st.markdown("### 🔬 AI-Powered Skin Analysis")
uploaded_file = st.file_uploader("Upload Skin Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # --- Run Prediction ---
    results = predict(image)
    top_class = max(results, key=results.get)
    confidence = results[top_class] * 100

    st.markdown("#### ✅ Prediction Result")
    st.success(f"**{top_class}** - {confidence:.1f}% Confidence")

    st.markdown("##### 📊 Class Probabilities")
    for cls, prob in results.items():
        st.write(f"**{cls}**: {prob * 100:.2f}%")
        st.progress(int(prob * 100))

    # --- Grad-CAM Heatmap ---
    st.markdown("##### 🔍 Grad-CAM Heatmap")

    # Load the model
    model = efficientnet_b0(pretrained=False)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 4)
    model.load_state_dict(torch.load("model/skin_classifier.pt", map_location="cpu"))
    model.eval()

    # Preprocess image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image.convert("RGB")).unsqueeze(0)

    # Generate Grad-CAM heatmap
    target_layer = model.features[-1]
    cam = GradCAM(model, target_layer)
    heatmap = cam.generate(input_tensor)

    # Overlay on image
    original_np = np.array(image.resize((224, 224)))
    if original_np.shape[2] == 4:  # Remove alpha channel if present
        original_np = original_np[:, :, :3]
    overlay = overlay_heatmap(heatmap, original_np, alpha=0.5)

    st.image(overlay, caption="Model Attention Heatmap", use_column_width=True)

    # --- Model Selection (for show) ---
    st.markdown("#### 🧠 Model Configuration")
    st.selectbox("Select AI Model", ["EfficientNet B0 (Active)"])
    st.write("EfficientNet B0 - Balanced accuracy and speed")
    st.button("🔍 Predict Again")

# --- Model Info ---
st.markdown("---")
st.markdown("## 📚 Model & Dataset Information")

st.markdown("### 📐 Model Architecture")
st.write("• Architecture: EfficientNet B0")
st.write("• Parameters: ~5M")
st.write("• Input Size: 224x224")
st.write("• Trained on: ISIC 2018")

st.markdown("### 📈 Performance Metrics")
st.write("• Accuracy: 88.2%")
st.write("• Precision: 0.837")
st.write("• Recall: 0.901")
st.write("• AUC: 0.923")

st.markdown("### ⚠️ Disclaimer")
st.warning("""
This is a research tool for educational/demo purposes only.
It does not replace professional medical advice or diagnosis.
""")

# --- Footer ---
st.markdown("---")
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    st.markdown("**SkinSight AI**")
    st.caption("AI-powered skin lesion analysis")
with col2:
    st.markdown("**Quick Links**")
    st.markdown("- [GitHub](#)\n- [About Us](#)")
with col3:
    st.markdown("**Contact**")
    st.markdown("📧 contact@skinsight.ai")

st.caption("© 2024 SkinSight AI | Made with ❤️ by Shreyash using Streamlit")



# import streamlit as st
# import torch
# from torchvision import transforms
# from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
# from PIL import Image
# import numpy as np
# import os

# from utils.gradcam import GradCAM, overlay_heatmap

# # 🔹 Load class names (same order as training)
# class_names = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']

# # 🔹 Load model
# @st.cache_resource
# def load_model():
#     weights = EfficientNet_B0_Weights.DEFAULT
#     model = efficientnet_b0(weights=None)
#     model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(class_names))
#     model.load_state_dict(torch.load("model/model.pth", map_location="cpu"))
#     model.eval()
#     return model

# model = load_model()
# target_layer = model.features[-1]  # last conv layer

# # 🔹 Preprocessing
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225]
#     )
# ])

# st.set_page_config(page_title="Skin Lesion Classifier", layout="centered")
# st.title("🧠 Skin Lesion Classifier with Grad-CAM")

uploaded_file = st.file_uploader("Upload a skin lesion image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    input_tensor = transform(image).unsqueeze(0)

    # 🔍 Grad-CAM
    gradcam = GradCAM(model, target_layer)
    heatmap = gradcam.generate(input_tensor)

    # 🔮 Prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1).squeeze().numpy()
        pred_idx = np.argmax(probs)
        pred_class = class_names[pred_idx]

    # 🖼️ Grad-CAM overlay
    image_np = np.array(image.resize((224, 224)))
    cam_overlay = overlay_heatmap(heatmap, image_np)

    # 🔠 Display prediction
    st.markdown(f"### 🏷️ Prediction: `{pred_class}`")
    st.markdown("### 🔥 Grad-CAM Heatmap:")
    st.image(cam_overlay, use_container_width=True)

    # 📊 Class Probability Bar Chart
    st.markdown("### 📊 Class Confidence Scores:")
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(class_names, probs, color='cornflowerblue')
    ax.set_ylabel("Confidence")
    ax.set_title("Model Prediction Confidence by Class")

    # Highlight predicted class
    bars[pred_idx].set_color('crimson')
    ax.text(pred_idx, probs[pred_idx] + 0.02, f"{probs[pred_idx]:.2f}",
            ha='center', va='bottom', fontweight='bold')

    st.pyplot(fig)
