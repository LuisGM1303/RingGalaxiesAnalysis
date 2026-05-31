import os
import random
from pathlib import Path
import numpy as np
import torch


def set_global_seed(seed: int = 42) -> None:
    """Fija semillas globales para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Operaciones determinísticas en CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    """Devuelve el dispositivo disponible (cuda|cpu)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_project_root() -> Path:
    """Resuelve la raíz del proyecto asumiendo que `src/` está dentro de la raíz."""
    return Path(__file__).resolve().parents[1]
