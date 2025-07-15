import torch
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)  # ✅ PyTorch ≥1.13

    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        target = output[0, class_idx]
        target.backward()

        # Compute Grad-CAM
        gradients = self.gradients[0]       # [C, H, W]
        activations = self.activations[0]   # [C, H, W]

        weights = gradients.mean(dim=(1, 2))  # Global average pooling

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / cam.max()
        cam = cam.cpu().numpy()

        cam = cv2.resize(cam, (224, 224))  # Resize to input size
        return cam

def overlay_heatmap(heatmap, image, alpha=0.4, colormap=cv2.COLORMAP_JET):
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), colormap)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    overlayed = cv2.addWeighted(heatmap_color, alpha, image_bgr, 1 - alpha, 0)
    return cv2.cvtColor(overlayed, cv2.COLOR_BGR2RGB)
