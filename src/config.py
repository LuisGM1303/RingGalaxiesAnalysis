from pathlib import Path

# Paths relativos al proyecto (asume estructura raíz con `src/`)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CATALOGS_DIR = RAW_DATA_DIR / "catalogs"
FITS_DIR = RAW_DATA_DIR / "legacy_fits_files"
CATALOG_PATH = CATALOGS_DIR / "classification_dataset.csv"

# Configuración global del experimento (valores por defecto)
GLOBAL_SEED = 13

CONFIG = {
    # Datos
    # NOTE: IMPORTANTE: 
    # Para astropy vitL16a el tamaño de imagen debe ser (518, 518) para evitar problemas de recorte
    "image_size": (518, 518),  # Cambia a (518, 518) si usas vitL16a
    "num_channels": 3,
    "num_classes": 2,

    # Entrenamiento
    "batch_size": 16,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "epochs": 10,

    # DataLoader
    "num_workers": 4,
    "pin_memory": True,

    # Reproducibilidad
    "seed": GLOBAL_SEED,

    # Transfer Learning
    "freeze_backbone": True,
}

BASELINE_CONFIG = {
    "model_name": "ResNet18",
    "pretrained": True,
    "pretrained_dataset": "ImageNet",
    "freeze_backbone": True,
    "fine_tuning_strategy": "Partial Fine-Tuning",
    "num_classes": 2,
    "input_channels": 3,
    "input_size": (256, 256),
}
