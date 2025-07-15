# app.py  ── drop this in AI-Powered-Medical-Image-Classifier/

import os, pickle, io
import streamlit as st
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
from utils.gradcam import GradCAM, overlay_heatmap

# ------------------------------------------------------------------
# 0.  CONFIG ────────────────────────────────────────────────────────
MODEL_PATH = "model/model.pth"          # make sure file exists here!
CLASSES    = ['Benign',
              'Melanoma',
              'Basal Cell Carcinoma',
              'Squamous Cell Carcinoma']   # edit if you have more/less

st.set_page_config(page_title="SkinSight AI", layout="centered")

# Optional logo
if os.path.exists("assets/logo.png"):
    st.image("assets/logo.png", width=140)
st.markdown("## :green[SkinSight AI – Instant Skin‑Lesion Diagnosis]")
st.write("Upload a skin image to get a prediction and a Grad‑CAM heat‑map.")

# ------------------------------------------------------------------
# 1.  AUTO‑DETECT & LOAD MODEL ─────────────────────────────────────
ARCH_CANDIDATES = {
    "efficientnet_b0": {
        "builder": lambda: __import__("torchvision.models"
                       ,fromlist=["efficientnet_b0"]).efficientnet_b0(pretrained=False),
        "target_layer": lambda m: m.features[-1],
        "fc_attr": ("classifier", 1)          # tuple → getattr + index
    },
    "resnet50": {
        "builder": lambda: __import__("torchvision.models"
                       ,fromlist=["resnet50"]).resnet50(pretrained=False),
        "target_layer": lambda m: m.layer4[-1],
        "fc_attr": ("fc", None)
    },
    "resnet34": {
        "builder": lambda: __import__("torchvision.models"
                       ,fromlist=["resnet34"]).resnet34(pretrained=False),
        "target_layer": lambda m: m.layer4[-1],
        "fc_attr": ("fc", None)
    },
    "vgg16": {
        "builder": lambda: __import__("torchvision.models"
                       ,fromlist=["vgg16"]).vgg16(pretrained=False),
        "target_layer": lambda m: m.features[-1],
        "fc_attr": ("classifier", 6)
    },
}

@st.cache_resource(show_spinner=False)
def load_state_dict(path:str):
    """Load state‑dict w/out re‑loading every run."""
    return torch.load(path, map_location="cpu")

def sniff_arch(state_dict):
    """Rudimentary key‑pattern sniffing."""
    keys = list(state_dict.keys())
    if any(k.startswith("classifier.1") for k in keys):
        return "efficientnet_b0"
    if any(k.startswith("layer4.") for k in keys) and "fc.weight" in keys:
        return "resnet50" if "layer3.5.conv1.weight" in keys else "resnet34"
    if any(k.startswith("features.") for k in keys) and "classifier.6.weight" in keys:
        return "vgg16"
    return None

@st.cache_resource(show_spinner=True)
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"⚠️ `{MODEL_PATH}` not found. Place your .pth file there.")

    raw_state = load_state_dict(MODEL_PATH)
    # Some training scripts save {"state_dict": …}
    state = raw_state["state_dict"] if isinstance(raw_state, dict) and \
             "state_dict" in raw_state else raw_state

    arch = sniff_arch(state)
    if arch is None or arch not in ARCH_CANDIDATES:
        raise ValueError("🚫 Couldn’t auto‑identify architecture. "
                         "Edit `ARCH_CANDIDATES` or set `arch` manually.")

    spec = ARCH_CANDIDATES[arch]
    model = spec["builder"]()
    # resize final FC layer to match #classes
    attr, idx = spec["fc_attr"]
    if idx is None:                   # e.g.  model.fc
        in_f = getattr(model, attr).in_features
        setattr(model, attr, torch.nn.Linear(in_f, len(CLASSES)))
    else:                             # e.g.  model.classifier[1]
        seq = getattr(model, attr)
        in_f = seq[idx].in_features
        seq[idx] = torch.nn.Linear(in_f, len(CLASSES))

    model.load_state_dict(state, strict=False)
    model.eval()
    return model, spec["target_layer"]

model, target_layer_fn = load_model()

# ------------------------------------------------------------------
# 2.  IMAGE PRE‑ & POST‑PROCESS ────────────────────────────────────
preprocess = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def predict_pil(img:Image.Image):
    x = preprocess(img.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()
    return probs

# ------------------------------------------------------------------
# 3.  UI – UPLOAD, PREDICT, GRAD‑CAM ───────────────────────────────
uploaded = st.file_uploader("Upload JPG/PNG", type=["jpg","jpeg","png"])
if uploaded:
    pil_img = Image.open(uploaded)
    st.image(pil_img, caption="Original upload", use_column_width=True)

    probs = predict_pil(pil_img)
    top_idx = int(np.argmax(probs))
    st.success(f"**{CLASSES[top_idx]}** – {probs[top_idx]*100:0.1f}% confidence")

    st.markdown("#### Class Probabilities")
    for lbl, p in zip(CLASSES, probs):
        st.write(f"{lbl}: {p*100:0.2f}%")
        st.progress(int(p*100))

    # ---- Grad‑CAM
    st.markdown("#### Grad‑CAM Heat‑map")
    cam = GradCAM(model, target_layer_fn(model))
    heat = cam.generate(preprocess(pil_img).unsqueeze(0))
    base_np = np.array(pil_img.resize((224,224)))
    if base_np.shape[2]==4: base_np = base_np[:,:,:3]
    overlay = overlay_heatmap(heat, base_np, alpha=0.45)
    st.image(overlay, caption="Model attention", use_column_width=True)

# ------------------------------------------------------------------
# 4.  FOOTER & INFO ────────────────────────────────────────────────
st.markdown("---")
st.caption("© 2025 SkinSight AI | Built with Streamlit + PyTorch  •  "
          )


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

# uploaded_file = st.file_uploader("Upload a skin lesion image", type=["jpg", "png", "jpeg"])

# if uploaded_file:
#     image = Image.open(uploaded_file).convert("RGB")
#     st.image(image, caption="Uploaded Image", use_column_width=True)

#     input_tensor = transform(image).unsqueeze(0)
    
#     # 🔍 Grad-CAM
#     gradcam = GradCAM(model, target_layer)
#     heatmap = gradcam.generate(input_tensor)

#     # 🔮 Prediction
#     with torch.no_grad():
#         outputs = model(input_tensor)
#         probs = torch.softmax(outputs, dim=1).squeeze().numpy()
#         pred_idx = np.argmax(probs)
#         pred_class = class_names[pred_idx]

#     # 🖼️ Overlay heatmap
#     image_np = np.array(image.resize((224, 224)))
#     cam_overlay = overlay_heatmap(heatmap, image_np)

#     st.markdown(f"### 🏷️ Prediction: `{pred_class}`")
#     st.markdown("### 🔥 Grad-CAM Heatmap:")
#     st.image(cam_overlay, use_column_width=True)

#     st.markdown("### 📊 Class Probabilities:")
#     for i, prob in enumerate(probs):
#         st.write(f"{class_names[i]}: `{prob:.2f}`")
