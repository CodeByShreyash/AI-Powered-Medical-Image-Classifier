import streamlit as st
import torch
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
import numpy as np
import os

from utils.gradcam import GradCAM, overlay_heatmap

# 🔹 Load class names (same order as training)
class_names = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']

# 🔹 Load model
@st.cache_resource
def load_model():
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(class_names))
    model.load_state_dict(torch.load("model/model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()
target_layer = model.features[-1]  # last conv layer

# 🔹 Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

st.set_page_config(page_title="Skin Lesion Classifier", layout="centered")
st.title("🧠 Skin Lesion Classifier with Grad-CAM")

uploaded_file = st.file_uploader("Upload a skin lesion image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

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

    # 🖼️ Overlay heatmap
    image_np = np.array(image.resize((224, 224)))
    cam_overlay = overlay_heatmap(heatmap, image_np)

    st.markdown(f"### 🏷️ Prediction: `{pred_class}`")
    st.markdown("### 🔥 Grad-CAM Heatmap:")
    st.image(cam_overlay, use_column_width=True)

    st.markdown("### 📊 Class Probabilities:")
    for i, prob in enumerate(probs):
        st.write(f"{class_names[i]}: `{prob:.2f}`")
