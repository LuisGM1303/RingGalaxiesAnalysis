import torch
import matplotlib.pyplot as plt
from pathlib import Path
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def generate_gradcam_visualization(model, dataset, device, target_label=1, save_path=None):
    if hasattr(model, "features"):
        target_layer = model.features[-1]
    elif hasattr(model, "layer4"):
        target_layer = model.layer4[-1]
    else:
        raise RuntimeError("No se encontró una capa compatible para GradCAM en el modelo.")

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

    output = model(input_tensor)
    prediction = int(torch.argmax(output, dim=1).item())

    targets = [ClassifierOutputTarget(prediction)]
    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    rgb_img = sample_tensor.permute(1, 2, 0).cpu().numpy()
    rgb_img = rgb_img - rgb_img.min()
    rgb_img = rgb_img / (rgb_img.max() + 1e-9)

    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    if save_path is not None:
        save_path = Path(save_path)
        plt.imsave(save_path, visualization)
        return save_path

    return visualization

