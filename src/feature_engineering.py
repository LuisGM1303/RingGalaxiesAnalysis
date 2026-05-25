from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from astropy.io import fits
from astropy.visualization import AsinhStretch
from scipy.ndimage import gaussian_filter
from skimage import exposure
from skimage.filters import unsharp_mask
from skimage.restoration import denoise_bilateral, denoise_wavelet


def load_fits(galaxy_id: str, parent_folder_path: str, label: int) -> Dict[str, np.ndarray]:
    """Carga los archivos FITS de una galaxia en las bandas g, r y z."""
    base_path = Path(parent_folder_path) / f"label_{label}"
    images: Dict[str, np.ndarray] = {}

    for band in ("g", "r", "z"):
        file_path = base_path / band / f"{galaxy_id}_{band}.fits"
        with fits.open(file_path) as hdul:
            data = hdul[0].data.astype(np.float64)
        images[band] = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    return images


def percentile_normalize(data: np.ndarray, lower: float = 0.5, upper: float = 99.5) -> np.ndarray:
    """Recorta los valores extremos y normaliza la imagen a [0, 1]."""
    v_min, v_max = np.percentile(data, [lower, upper])
    data = np.clip(data, v_min, v_max)
    denom = v_max - v_min
    return (data - v_min) / denom if denom != 0 else np.zeros_like(data)


def unsharp_masking(data: np.ndarray, radius: float = 6.0, amount: float = 3.0) -> np.ndarray:
    """Realza estructuras locales con unsharp masking."""
    enhanced = unsharp_mask(data, radius=radius, amount=amount, preserve_range=True)
    return np.clip(enhanced, 0.0, 1.0)


def asinh_transform(data: np.ndarray, a: float = 0.1) -> np.ndarray:
    """Aplica la transformación Asinh al rango dinámico."""
    stretch = AsinhStretch(a=a)
    transformed = stretch(data)
    transformed = transformed - transformed.min()
    return transformed / transformed.max() if transformed.max() != 0 else transformed


def apply_clahe(data: np.ndarray, clip_limit: float = 0.01) -> np.ndarray:
    """Mejora el contraste local usando CLAHE."""
    return exposure.equalize_adapthist(data, clip_limit=clip_limit)


def gaussian_denoise(data: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Suavizado gaussiano ligero para reducir ruido."""
    return gaussian_filter(data, sigma=sigma)


def bilateral_denoise(data: np.ndarray, sigma_color: float = 0.05, sigma_spatial: float = 3) -> np.ndarray:
    """Suavizado bilateral para preservar contornos."""
    return denoise_bilateral(data, sigma_color=sigma_color, sigma_spatial=sigma_spatial, channel_axis=None)


def wavelet_denoise(data: np.ndarray) -> np.ndarray:
    """Denoising basado en wavelets."""
    return denoise_wavelet(data, method="BayesShrink", mode="soft", rescale_sigma=True)


def residual_map(data: np.ndarray, processed_data: np.ndarray) -> np.ndarray:
    """Calcula el mapa de residuos entre una imagen original y su versión procesada."""
    return data - processed_data


def feature_engineering_pipeline(data: np.ndarray, target_shape: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """Pipeline integrado de preprocesamiento por banda."""
    clean_data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    normalized = percentile_normalize(clean_data)
    normalized = np.clip(normalized, 0.0, 1.0)
    denoised = gaussian_denoise(normalized, sigma=1.0)
    sharpened = unsharp_masking(denoised, radius=6.0, amount=3.0)
    stretched = asinh_transform(sharpened, a=0.1)

    if stretched.shape != target_shape:
        from skimage.transform import resize

        stretched = resize(stretched, target_shape, anti_aliasing=True)

    return stretched
