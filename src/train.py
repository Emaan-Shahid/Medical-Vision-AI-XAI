import time
import torch
import torch.nn as nn
import pandas as pd
from torchvision import models


def build_model(name: str, num_classes: int = 4):
    """Load a model pre-trained on ImageNet and replace its last layer."""
    if name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        head = model.fc
    elif name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        head = model.classifier
    else:
        raise ValueError(f'Unknown model: {name}')
    return model, head


def set_backbone_trainable(model, head, trainable: bool):
    """Stage 1: trainable=False (only the new last layer learns). Stage 2: True."""
    for p in model.parameters():
        p.requires_grad = trainable
    for p in head.parameters():
        p.requires_grad = True


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    """One pass over a loader. Learns if an optimizer is given, otherwise only grades."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type='cuda', enabled=(device == 'cuda')):
                outputs = model(images)
                loss = criterion(outputs, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            n += images.size(0)
    return total_loss / n, correct / n


def fit(model, head, loaders, class_weights, device, name, save_dir, results_dir,
        stage1_epochs=3, stage2_epochs=12, patience=4):
    """Two-stage training. Saves the best model (lowest validation loss) to Drive."""
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.1)
    scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda'))
    best_val, best_path = float('inf'), f'{save_dir}/{name}_best.pt'
    history = []

    stages = [(1, stage1_epochs, 1e-3, False),   # (stage, epochs, learning rate, unfreeze?)
              (2, stage2_epochs, 1e-4, True)]
    for stage, epochs, lr, unfreeze in stages:
        if epochs == 0:
            continue
        set_backbone_trainable(model, head, unfreeze)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        bad_epochs = 0

        for ep in range(1, epochs + 1):
            t0 = time.time()
            tr_loss, tr_acc = run_epoch(model, loaders['train'], criterion, device, optimizer, scaler)
            va_loss, va_acc = run_epoch(model, loaders['val'], criterion, device)
            scheduler.step()
            secs = time.time() - t0

            saved = ''
            if va_loss < best_val:
                best_val, bad_epochs = va_loss, 0
                torch.save(model.state_dict(), best_path)
                saved = '  <- saved best'
            else:
                bad_epochs += 1

            history.append(dict(stage=stage, epoch=ep, train_loss=tr_loss, train_acc=tr_acc,
                                val_loss=va_loss, val_acc=va_acc, seconds=secs))
            print(f'Stage {stage} | epoch {ep:2d}/{epochs} | '
                  f'train loss {tr_loss:.3f} acc {tr_acc:.3f} | '
                  f'val loss {va_loss:.3f} acc {va_acc:.3f} | {secs:.0f}s{saved}')

            if stage == 2 and bad_epochs >= patience:
                print(f'Early stopping: no improvement for {patience} epochs.')
                break

    model.load_state_dict(torch.load(best_path, map_location=device))   # go back to the best version
    hist = pd.DataFrame(history)
    hist.to_csv(f'{results_dir}/{name}_history.csv', index=False)
    return model, hist
