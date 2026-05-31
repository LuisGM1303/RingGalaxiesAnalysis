import torch
import numpy as np
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def generate_gradcam_visualization(model, dataset, device, target_label=1):
    """Genera y muestra un Grad-CAM para una muestra del `dataset` con `target_label`."""
    # Selección de la capa objetivo (compatible con ResNet)
    if hasattr(model, "features"):
        target_layer = model.features[-1]
    elif hasattr(model, "layer4"):
        target_layer = model.layer4[-1]
    else:
        raise RuntimeError("No se encontró una capa compatible para GradCAM en el modelo.")

    # Buscar una muestra con la etiqueta objetivo
    target_sample = None
    for img, lbl in dataset:
        if int(lbl) == int(target_label):
            target_sample = (img, lbl)
            break

    if target_sample is None:
        raise ValueError("No se encontró una muestra con la etiqueta objetivo en el dataset.")

    sample_tensor, sample_label = target_sample
    input_tensor = sample_tensor.unsqueeze(0).to(device)

    model.eval()
    model.requires_grad_(True)

    with torch.no_grad():
        output = model(input_tensor)
        prediction = int(torch.argmax(output, dim=1).item())

    targets = [ClassifierOutputTarget(prediction)]

    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    grayscale_cam = grayscale_cam[0]

    # Preparar imagen para visualización
    rgb_img = sample_tensor.permute(1, 2, 0).cpu().numpy()
    rgb_img = rgb_img - rgb_img.min()
    rgb_img = rgb_img / (rgb_img.max() + 1e-9)

    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    plt.figure(figsize=(8, 8))
    plt.imshow(visualization)
    plt.title(f"Real: {sample_label} | Pred: {prediction}")
    plt.axis("off")
    plt.show()
