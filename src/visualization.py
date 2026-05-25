import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import make_lupton_rgb


def plot_galaxy(galaxy_id: str, images: dict[str, np.ndarray], label: int = 1) -> None:
    """Muestra las bandas g, r y z de una galaxia en escala de grises."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, band in zip(axes, ("g", "r", "z")):
        ax.imshow(images[band], cmap="gray", norm=mcolors.Normalize())
        ax.set_title(f"Galaxia {galaxy_id} (label={label}) - Banda {band}")
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def plot_rgb(rgb_image: np.ndarray) -> None:
    """Muestra una imagen RGB generada con el algoritmo de Lupton."""
    plt.figure(figsize=(6, 6))
    plt.imshow(rgb_image)
    plt.title("Imagen RGB combinada (g, r, z)")
    plt.axis("off")
    plt.show()


def make_lupton_image(g_images: dict[str, np.ndarray], Q: float = 10.0, stretch: float = 0.5) -> np.ndarray:
    """Genera una imagen RGB con Lupton a partir de las bandas g, r y z."""
    return make_lupton_rgb(
        g_images["g"],
        g_images["r"],
        g_images["z"],
        Q=Q,
        stretch=stretch,
    )
