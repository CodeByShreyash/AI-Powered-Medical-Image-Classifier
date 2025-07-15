# utils/predict.py

import torch
from torchvision import transforms
from PIL import Image

# Define the classes
CLASSES = ['Benign', 'Melanoma', 'Basal Cell Carcinoma', 'Squamous Cell Carcinoma']

# Load the model
def load_model():
    from torchvision.models import efficientnet_b0
    model = efficientnet_b0(pretrained=False)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(CLASSES))
    # model.load_state_dict(torch.load('model/skin_classifier.pt', map_location='cpu'))
    model.load_state_dict(torch.load('model/model.pth', map_location='cpu'))

    model.eval()
    return model

# Preprocess the uploaded image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image.convert("RGB")).unsqueeze(0)

# Predict function
def predict(image):
    model = load_model()
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0).numpy()
    return dict(zip(CLASSES, probabilities))
