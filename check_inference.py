from PIL import Image
from src.inference import load_model, predict_with_cam, CLASSES

for name in ['resnet50', 'efficientnet_b0']:
    model = load_model(name)
    for f in ['glioma_1', 'notumor_1', 'glioma_MISSED_1']:
        img = Image.open(f'app/samples/{f}.jpg')
        cam, k, probs = predict_with_cam(model, name, img)
        print(f'{name:16s} {f:16s} -> {CLASSES[k]:11s} conf {probs[k]:.2f} | heatmap {cam.shape}, max {cam.max():.2f}')