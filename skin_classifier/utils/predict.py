import torch
from torchvision import transforms
from torchvision.models import efficientnet_b0, resnet18
from PIL import Image

# Define the consistent class labels
CLASSES = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']

# --- Load either model based on input ---
def load_model(model_type="efficientnet"):
    if model_type == "efficientnet":
        model = efficientnet_b0(weights=None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(CLASSES))
        model.load_state_dict(torch.load('model/efficientnet.pth', map_location='cpu'))
    elif model_type == "resnet18":
        model = resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, len(CLASSES))
        model.load_state_dict(torch.load('model/resnet18.pth', map_location='cpu'))
    else:
        raise ValueError("Unsupported model type. Choose 'efficientnet' or 'resnet18'.")

    model.eval()
    return model

# --- Preprocessing for input image ---
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image.convert("RGB")).unsqueeze(0)

# --- Prediction utility ---
def predict(image, model_type="efficientnet"):
    model = load_model(model_type)
    input_tensor = preprocess_image(image)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0).numpy()

    return dict(zip(CLASSES, probabilities))
