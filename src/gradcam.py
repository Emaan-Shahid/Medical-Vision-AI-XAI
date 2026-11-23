import torch
import torch.nn.functional as F


class GradCAM:
    """Grad-CAM written from scratch.

    1. Run the picture through the model, remembering the output of one layer (activations).
    2. Ask the model how the chosen class score depends on that layer (gradients).
    3. Average the gradients per channel -> one importance weight per channel.
    4. Weighted sum of the channels, keep only positive evidence (ReLU), stretch to 224x224.
    """

    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.acts, self.grads = None, None
        self.handle = target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output):
        self.acts = output.detach()
        output.register_hook(lambda g: setattr(self, "grads", g.detach()))

    def __call__(self, x, class_idx=None):
        """x: one picture, shape [1, 3, 224, 224]. Returns heatmap, class used, probabilities."""
        self.model.zero_grad()
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(1))
        logits[0, class_idx].backward()

        weights = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        probs = torch.softmax(logits.detach(), dim=1)[0]
        return cam.cpu().numpy(), class_idx, probs.cpu().numpy()

    def close(self):
        self.handle.remove()


def target_layer_for(model, name):
    """The last convolutional block of each model."""
    if name == "resnet50":
        return model.layer4[-1]
    if name == "efficientnet_b0":
        return model.features[-1]
    raise ValueError(name)
