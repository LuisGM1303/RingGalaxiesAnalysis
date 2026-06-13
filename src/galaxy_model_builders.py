"""
galaxy_model_builders.py
========================
Seis funciones de construcción de modelos para clasificación binaria de galaxias,
todas con la misma firma y contrato de retorno que build_resnet18 del notebook:

    (model: nn.Module, trainable_params: int, frozen_params: int)

Arquitecturas incluidas
-----------------------
BLOQUE A — Modelos preentrenados en galaxias (mínimo domain shift)
  1. build_zoobot_convnext      ConvNeXt preentrenado en ~10M galaxias GZ (Zoobot 2.0)
  2. build_astroclip_vit        ViT-L/16 preentrenado en 76M imágenes DESI g,r,z (AstroCLIP)

BLOQUE B — Transformers de visión preentrenados en ImageNet (transfer learning moderno)
  3. build_vit_base             ViT-B/16 de Google/HuggingFace (ImageNet-21k → ImageNet-1k)
  4. build_swin_transformer     Swin-T de Microsoft via timm (ImageNet)

BLOQUE C — CNNs clásicas probadas en galaxias con fine-tuning
  5. build_efficientnet_b5      EfficientNet-B5 (ImageNet): top-3 en Galaxy Zoo 2 Kaggle
  6. build_resnet50             ResNet-50 (ImageNet): backbone estándar en astrofísica comp.

Dependencias
------------
    pip install timm transformers         # para ViT, Swin, EfficientNet, AstroCLIP
    pip install "zoobot[pytorch]"         # solo para build_zoobot_convnext
    pip install torchvision               # para ResNet-50

Referencias
-----------
  Zoobot 2.0    — Walmsley et al. 2023  https://arxiv.org/abs/2404.02973
  AstroCLIP     — Parker et al. 2024   https://arxiv.org/abs/2310.03024
  ViT galaxias  — Lin et al. 2021      https://arxiv.org/abs/2110.01024
  CvT galaxias  — A&A 2024            https://doi.org/10.1051/0004-6361/202348544
  EfficientNet  — Kalvankar et al. 2021 https://arxiv.org/abs/2008.13611
  Swin+GalNet   — ICCS 2023           (SwinT Galnet, parámetros regression)
"""

from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Utilidad: contador de parámetros (idéntico al notebook)
# ---------------------------------------------------------------------------

def _count_params(model: nn.Module) -> Tuple[int, int]:
    trainable, frozen = 0, 0
    for p in model.parameters():
        n = p.numel()
        if p.requires_grad:
            trainable += n
        else:
            frozen += n
    return trainable, frozen


def _freeze_backbone(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False


def _unfreeze_head(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = True

def _register_gradcam(
    model: nn.Module,
    target_layer,
    reshape_transform=None,
):
    model.gradcam_target_layer = target_layer
    model.gradcam_reshape_transform = reshape_transform

def get_gradcam_config(model):

    if hasattr(model, "gradcam_target_layer"):
        return (
            model.gradcam_target_layer,
            getattr(model, "gradcam_reshape_transform", None),
        )

    raise RuntimeError(
        "El modelo no tiene configuración GradCAM registrada"
    )   

# ===========================================================================
# BLOQUE A — Modelos preentrenados directamente en imágenes de galaxias
# ===========================================================================

# ---------------------------------------------------------------------------
# 1. Zoobot 2.0 — ConvNeXt preentrenado en Galaxy Zoo (~10M galaxias)
# ---------------------------------------------------------------------------

class _ZoobotWrapper(nn.Module):
    """nn.Module puro que envuelve el encoder timm de Zoobot con una cabeza
    clasificadora equivalente a la de build_resnet18."""

    def __init__(self, encoder: nn.Module, encoder_dim: int,
                 num_classes: int, head_hidden: int, head_dropout: float):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(encoder_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(p=head_dropout),
            nn.Linear(head_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


def build_zoobot_convnext(
    num_classes: int = 2,
    variant: str = "convnext_nano",
    freeze_backbone: bool = True,
    head_hidden: int = 128,
    head_dropout: float = 0.3,
    greyscale: bool = False,
    device: torch.device | None = None,
) -> Tuple[nn.Module, int, int]:
    """
    ConvNeXt preentrenado con Zoobot 2.0 sobre ~10M galaxias de Galaxy Zoo.
    Es el modelo con menor domain shift para imágenes astronómicas.

    Variantes disponibles (en orden creciente de tamaño/VRAM):
        convnext_pico  (encoder_dim≈96)
        convnext_nano  (encoder_dim≈256)  ← DEFAULT
        convnext_tiny  (encoder_dim≈768)
        convnext_small (encoder_dim≈768)
        convnext_base  (encoder_dim≈1024)
        convnext_large (encoder_dim≈1536)

    Instalación:  pip install "zoobot[pytorch]"
    Referencia:   Walmsley et al. 2023 (https://arxiv.org/abs/2404.02973)
    Pesos:        https://huggingface.co/collections/mwalmsley/zoobot-encoders-65fa14ae92911b173712b874
    """
    try:
        import timm
    except ImportError:
        raise ImportError("Ejecuta: pip install 'zoobot[pytorch]'  (instala timm)")

    hf_name = f"hf_hub:mwalmsley/zoobot-encoder-{variant}"
    timm_kwargs = {"num_classes": 0, "pretrained": True}
    if greyscale:
        timm_kwargs["in_chans"] = 1

    encoder = timm.create_model(hf_name, **timm_kwargs)
    encoder_dim: int = encoder.num_features

    model = _ZoobotWrapper(encoder, encoder_dim, num_classes,
                           head_hidden, head_dropout)
    _register_gradcam(
    model,
    target_layer=model.encoder.stages[-1].blocks[-1],
    reshape_transform=None,
)
    if freeze_backbone:
        _freeze_backbone(model.encoder)
    _unfreeze_head(model.head)

    trainable, frozen = _count_params(model)
    if device is not None:
        model = model.to(device)
    return model, trainable, frozen


# ---------------------------------------------------------------------------
# 2. AstroCLIP — ViT-L/16 preentrenado en 76M imágenes DESI (g,r,z)
# ---------------------------------------------------------------------------

class _AstroCLIPWrapper(nn.Module):
    """Envuelve el image encoder de AstroCLIP (ViT-L) con cabeza clasificadora."""

    def __init__(self, encoder: nn.Module, encoder_dim: int,
                 num_classes: int, head_hidden: int, head_dropout: float):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(encoder_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(p=head_dropout),
            nn.Linear(head_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # AstroCLIP usa timm ViT con forward_features → pooled CLS token
        feats = self.encoder.forward_features(x)          # (B, seq, dim)
        if feats.ndim == 3:
            feats = feats[:, 0, :]                         # CLS token → (B, dim)
        return self.head(feats)


def build_astroclip_vit(
    num_classes: int = 2,
    freeze_backbone: bool = True,
    head_hidden: int = 128,
    head_dropout: float = 0.3,
    device: torch.device | None = None,
) -> Tuple[nn.Module, int, int]:
    """
    ViT-L/14 (AstroDINO) preentrenado en 76M imágenes DESI g,r,z.
    Pesos públicos: hf.co/polymathic-ai/astrodino  (astrodino.ckpt, 1.31 GB)

    El checkpoint es un archivo PyTorch Lightning; se extrae el encoder
    DINOv2 ViT-L/14 directamente del state_dict.

    Referencia: Parker et al. 2024, MNRAS 531 4990, arXiv:2310.03024
    """
    try:
        from huggingface_hub import hf_hub_download
        import timm
    except ImportError:
        raise ImportError("Ejecuta: pip install timm huggingface_hub")

    # 1. Descargar checkpoint desde HuggingFace Hub
    ckpt_path = hf_hub_download(
        repo_id="polymathic-ai/astrodino",
        filename="astrodino.ckpt",
    )

    # 2. Cargar state_dict del checkpoint Lightning
    ckpt = torch.load(ckpt_path, map_location="cpu")
    # Los checkpoints Lightning guardan pesos bajo "state_dict"
    state_dict = ckpt.get("state_dict", ckpt)

    # 3. Crear arquitectura base: DINOv2 ViT-L/14 via timm
    #    AstroDINO usa exactamente esta arquitectura con in_chans=3
    encoder = timm.create_model(
        "vit_large_patch14_dinov2",
        pretrained=False,     # NO cargar pesos ImageNet, usamos los astronómicos
        num_classes=0,        # sin cabeza de clasificación
        global_pool="token",  # CLS token
    )

    # 4. Filtrar y cargar solo las claves del encoder
    #    En el ckpt Lightning, las claves del encoder empiezan con "model."
    encoder_sd = {
        k.replace("model.", "", 1): v
        for k, v in state_dict.items()
        if k.startswith("model.")
    }
    missing, unexpected = encoder.load_state_dict(encoder_sd, strict=False)
    if missing:
        print(f"[INFO] Claves faltantes (esperado si hay cabeza): {len(missing)}")

    encoder_dim: int = encoder.num_features  # 1024 para ViT-L

    model = _AstroCLIPWrapper(encoder, encoder_dim, num_classes,
                              head_hidden, head_dropout)

    if freeze_backbone:
        _freeze_backbone(model.encoder)
    _unfreeze_head(model.head)

    trainable, frozen = _count_params(model)
    if device is not None:
        model = model.to(device)
    return model, trainable, frozen


# ===========================================================================
# BLOQUE B — Vision Transformers preentrenados en ImageNet
# ===========================================================================

# ---------------------------------------------------------------------------
# 3. ViT-B/16 (Google) — primer transformer aplicado a galaxias (Lin et al. 2021)
# ---------------------------------------------------------------------------

def build_vit_base(
    num_classes: int = 2,
    freeze_backbone: bool = True,
    head_hidden: int = 128,
    head_dropout: float = 0.3,
    img_size: int = 224,
    device: torch.device | None = None,
) -> Tuple[nn.Module, int, int]:
    """
    Vision Transformer ViT-B/16 preentrenado en ImageNet-21k → fine-tuned ImageNet-1k.

    Lin et al. (2021) fue el PRIMER trabajo en aplicar ViT a clasificación morfológica
    de galaxias, demostrando que es especialmente bueno con galaxias pequeñas y tenues
    —exactamente el tipo de objetos difíciles en tu problema (anillos internos débiles).
    Posteriormente, CvT-13 (Convolutional vision Transformer, A&A 2024) superó a ViT-B/16
    con >98% accuracy en Galaxy Zoo, mostrando el potencial del paradigma transformer.

    Instalación:  pip install transformers
    Referencia:   Lin et al. 2021 (https://arxiv.org/abs/2110.01024)
                  A&A 2024 CvT galaxias (doi:10.1051/0004-6361/202348544)
    Pesos:        google/vit-base-patch16-224-in21k (HuggingFace Hub)

    Nota: requiere imágenes de 224×224 píxeles.
    """
    try:
        from transformers import ViTModel
    except ImportError:
        raise ImportError("Ejecuta: pip install transformers")

    vit = ViTModel.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        add_pooling_layer=True,     # activa pooled_output (CLS token)
    )
    encoder_dim = vit.config.hidden_size  # 768 para ViT-B

    class ViTClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = vit
            self.head = nn.Sequential(
                nn.Linear(encoder_dim, head_hidden),
                nn.ReLU(),
                nn.Dropout(p=head_dropout),
                nn.Linear(head_hidden, num_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # ViTModel espera pixel_values: (B, C, H, W)
            out = self.encoder(pixel_values=x)
            return self.head(out.pooler_output)  # (B, 768)

    model = ViTClassifier()

    if freeze_backbone:
        _freeze_backbone(model.encoder)
    _unfreeze_head(model.head)

    trainable, frozen = _count_params(model)
    if device is not None:
        model = model.to(device)
    return model, trainable, frozen


# ---------------------------------------------------------------------------
# 4. Swin Transformer-Tiny (Microsoft) — atención local con ventanas deslizantes
# ---------------------------------------------------------------------------

def build_swin_transformer(
    num_classes: int = 2,
    variant: str = "swin_tiny_patch4_window7_224",
    freeze_backbone: bool = True,
    head_hidden: int = 128,
    head_dropout: float = 0.3,
    device: torch.device | None = None,
) -> Tuple[nn.Module, int, int]:
    """
    Swin Transformer preentrenado en ImageNet via timm.

    A diferencia de ViT (atención global), Swin usa atención local con ventanas
    deslizantes (shifted windows), lo que le da mejor inductive bias espacial —
    útil para detectar estructuras morfológicas localizadas como anillos internos.
    SwinT Galnet (ICCS 2023) demostró mejoras sobre redes CNN puras en tareas de
    regresión de parámetros galácticos.

    Variantes disponibles (timm):
        'swin_tiny_patch4_window7_224'    (28M params)  ← DEFAULT
        'swin_small_patch4_window7_224'   (50M params)
        'swin_base_patch4_window7_224'    (88M params)
        'swin_large_patch4_window12_384'  (197M params, img 384×384)

    Instalación:  pip install timm
    Referencia:   Liu et al. 2021 Swin, ICCS 2023 SwinT Galnet
    Pesos:        ImageNet-1k / ImageNet-22k según variante (timm hub)
    """
    try:
        import timm
    except ImportError:
        raise ImportError("Ejecuta: pip install timm")

    encoder = timm.create_model(variant, pretrained=True, num_classes=0)
    encoder_dim: int = encoder.num_features  # 768 para swin_tiny

    class SwinClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            self.head = nn.Sequential(
                nn.Linear(encoder_dim, head_hidden),
                nn.ReLU(),
                nn.Dropout(p=head_dropout),
                nn.Linear(head_hidden, num_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.head(self.encoder(x))

    model = SwinClassifier()

    if freeze_backbone:
        _freeze_backbone(model.encoder)
    _unfreeze_head(model.head)

    trainable, frozen = _count_params(model)
    if device is not None:
        model = model.to(device)
    return model, trainable, frozen


# ===========================================================================
# BLOQUE C — CNNs clásicas validadas en morfología galáctica
# ===========================================================================

# ---------------------------------------------------------------------------
# 5. EfficientNet-B5 — top-3 en Galaxy Zoo 2 Kaggle, >96% en clasificación GZ
# ---------------------------------------------------------------------------

def build_efficientnet_b5(
    num_classes: int = 2,
    freeze_backbone: bool = True,
    head_hidden: int = 128,
    head_dropout: float = 0.3,
    device: torch.device | None = None,
) -> Tuple[nn.Module, int, int]:
    """
    EfficientNet-B5 preentrenado en ImageNet, fine-tuneado para galaxias.

    Kalvankar et al. (2021) aplicaron EfficientNet-B5 al Galaxy Zoo 2 Kaggle
    (79,975 galaxias) y obtuvieron top-3 en el leaderboard público. Estudios
    de 2024 (A&A, MDPI) confirman >96% accuracy en clasificación morfológica
    de galaxias usando fine-tuning sobre ImageNet weights. Supera a ResNet-26
    y ResNet-50 en precisión y recall en clasificación de morfologías galácticas.

    Instalación:  pip install timm
    Referencia:   Kalvankar et al. 2021 (https://arxiv.org/abs/2008.13611)
                  A&A 2024 CvT comparison table (doi:10.1051/0004-6361/202348544)
    Pesos:        ImageNet (timm, noisy-student pretraining de Google)
    """
    try:
        import timm
    except ImportError:
        raise ImportError("Ejecuta: pip install timm")

    encoder = timm.create_model("efficientnet_b5", pretrained=True, num_classes=0)
    encoder_dim: int = encoder.num_features  # 2048 para B5

    class EfficientNetClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            self.head = nn.Sequential(
                nn.Linear(encoder_dim, head_hidden),
                nn.ReLU(),
                nn.Dropout(p=head_dropout),
                nn.Linear(head_hidden, num_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.head(self.encoder(x))

    model = EfficientNetClassifier()
 
    _register_gradcam(
        model,
        target_layer=model.encoder.blocks[-1],
        reshape_transform=None,
    )
   
    if freeze_backbone:
        _freeze_backbone(model.encoder)
    _unfreeze_head(model.head)

    trainable, frozen = _count_params(model)
    if device is not None:
        model = model.to(device)
    return model, trainable, frozen


# ---------------------------------------------------------------------------
# 6. ResNet-50 — backbone estándar en astrofísica computacional
# ---------------------------------------------------------------------------

def build_resnet50(
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_backbone: bool = True,
    head_hidden: int = 128,
    head_dropout: float = 0.3,
    device: torch.device | None = None,
) -> Tuple[nn.Module, int, int]:
    """
    ResNet-50 preentrenado en ImageNet (torchvision).

    Aunque ResNet-18 ya es el baseline del notebook, ResNet-50 es una arquitectura
    distinta con más capacidad representacional (25M vs 11M parámetros, bloques
    Bottleneck vs BasicBlock). Es el backbone más citado en transfer learning
    astronómico y sirve como punto de comparación "más fuerte dentro de ImageNet"
    frente a los modelos especializados en galaxias.

    Instalación:  pip install torchvision  (ya disponible en el notebook)
    Referencia:   Domínguez Sánchez et al. 2019 (transfer survey→survey)
                  Variawa et al. 2022 (ResNet50 fine-tuned Galaxy Zoo 2 + EFIGI)
    Pesos:        ImageNet-1k (torchvision ResNet50_Weights.DEFAULT)
    """
    from torchvision.models import resnet50, ResNet50_Weights

    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = resnet50(weights=weights)
    _register_gradcam(
        model,
        target_layer=model.layer4[-1],
        reshape_transform=None,
    )
    if freeze_backbone:
        _freeze_backbone(model)

    in_features = model.fc.in_features  # 2048
    model.fc = nn.Sequential(
        nn.Linear(in_features, head_hidden),
        nn.ReLU(),
        nn.Dropout(p=head_dropout),
        nn.Linear(head_hidden, num_classes),
    )
    _unfreeze_head(model.fc)

    trainable, frozen = _count_params(model)
    if device is not None:
        model = model.to(device)
    return model, trainable, frozen


# ===========================================================================
# Tabla de comparación — resumen de las 6 arquitecturas
# ===========================================================================

def print_architecture_summary() -> None:
    """Imprime una tabla comparativa de las seis arquitecturas."""

    rows = [
        # (nombre, preentren., dominio, tipo, domain_shift, disponibilidad)
        ("1. Zoobot ConvNeXt-nano",  "~10M galaxias GZ",     "Galaxias",  "CNN híbrida",     "Mínimo ✅",  "pip zoobot"),
        ("2. AstroCLIP ViT-L/16",   "76M imgs DESI g,r,z",  "Galaxias",  "Transformer",     "Mínimo ✅",  "pip timm"),
        ("3. ViT-B/16 (Google)",     "ImageNet-21k",          "Natural",   "Transformer",     "Medio ⚠️",   "pip transformers"),
        ("4. Swin-T (Microsoft)",    "ImageNet-1k",           "Natural",   "Transformer",     "Medio ⚠️",   "pip timm"),
        ("5. EfficientNet-B5",       "ImageNet (noisy-stud)", "Natural",   "CNN escalada",    "Medio ⚠️",   "pip timm"),
        ("6. ResNet-50",             "ImageNet-1k",           "Natural",   "CNN residual",    "Alto ❌",    "torchvision"),
    ]

    header = f"{'Arquitectura':<28} {'Preentren.':<24} {'Dominio':<10} {'Tipo':<14} {'Domain Shift':<14} {'Install'}"
    print("\n" + "=" * 105)
    print("RESUMEN DE ARQUITECTURAS — clasificación binaria de galaxias")
    print("=" * 105)
    print(header)
    print("-" * 105)
    for r in rows:
        print(f"{r[0]:<28} {r[1]:<24} {r[2]:<10} {r[3]:<14} {r[4]:<14} {r[5]}")
    print("=" * 105)

    print("""
ESTRATEGIA EXPERIMENTAL RECOMENDADA
─────────────────────────────────────────────────────────────────────────────
Fase 1 — Baseline ya hecho:       ResNet-18 (ImageNet)
Fase 2 — CNNs clásicas fuertes:   ResNet-50 (6)  →  EfficientNet-B5 (5)
Fase 3 — Transformers ImageNet:   ViT-B/16  (3)  →  Swin-T           (4)
Fase 4 — Dominio astronómico:     Zoobot    (1)  →  AstroCLIP ViT-L  (2)

Hipótesis a validar:
  • ¿Los modelos de galaxias (1,2) superan a los de ImageNet (3-6)?
  • ¿Los transformers detectan mejor anillos internos que las CNNs?
  • ¿El domain shift es el factor más limitante, o lo es la capacidad del modelo?
─────────────────────────────────────────────────────────────────────────────
""")


if __name__ == "__main__":
    print_architecture_summary()
