import torch
import torch.nn as nn
from torchvision import models

from src.preprocessing import get_transforms
from src.gradcam import GradCAM, target_layer_for

CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']   # same order as training


def load_model(name, models_dir='models', device='cpu'):
    """Build the architecture without downloading weights, then load OUR trained weights."""
    if name == 'resnet50':
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 4)
    elif name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 4)
    else:
        raise ValueError(name)
    state = torch.load(f'{models_dir}/{name}_best.pt', map_location=device)
    model.load_state_dict(state)
    return model.to(device).eval()


def predict_with_cam(model, name, image, class_idx=None):
    """image: a PIL image. Returns (heatmap 224x224 in 0..1, class index used, 4 probabilities)."""
    x = get_transforms(train=False)(image.convert('RGB')).unsqueeze(0)
    cam_tool = GradCAM(model, target_layer_for(model, name))
    try:
        cam, k, probs = cam_tool(x, class_idx=class_idx)
    finally:
        cam_tool.close()
    return cam, k, probs