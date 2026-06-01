import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .dataset import GalaxyDataset


def load_catalog(catalog_path: Path, report_path: Path | None = None) -> pd.DataFrame:
    """Carga el catálogo y elimina columnas inválidas o filas corruptas."""
    catalog_path = Path(catalog_path)
    if not catalog_path.exists():
        raise FileNotFoundError(f"Catalog not found: {catalog_path}")

    df = pd.read_csv(catalog_path)
    df.drop(columns=["source"], errors="ignore", inplace=True)

    if report_path is not None and Path(report_path).exists():
        validation_report = pd.read_csv(report_path)
        invalid_ids = validation_report[validation_report["complete"] == False]["name"].tolist()
        df = df[~df["name"].isin(invalid_ids)].reset_index(drop=True)

    return df


def split_catalog(
    df: pd.DataFrame,
    seed: int,
    test_size: float = 0.30,
    label_column: str = "label",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Divide el catálogo en train/val/test de forma estratificada."""
    train_df, temp_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[label_column],
        random_state=seed,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df[label_column],
        random_state=seed,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def build_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    fits_dir: str | Path,
    config: dict,
) -> tuple[DataLoader, DataLoader, DataLoader, GalaxyDataset, GalaxyDataset, GalaxyDataset]:
    """Construye datasets y dataloaders para train, val y test."""
    train_dataset = GalaxyDataset(train_df, fits_dir, target_shape=config["image_size"], augment=True)
    val_dataset = GalaxyDataset(val_df, fits_dir, target_shape=config["image_size"], augment=False)
    test_dataset = GalaxyDataset(test_df, fits_dir, target_shape=config["image_size"], augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
    )

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset


def calculate_class_weights(
    train_df: pd.DataFrame,
    num_classes: int,
    device: torch.device,
    label_column: str = "label",
) -> torch.Tensor:
    """Calcula pesos de clase a partir del dataset de entrenamiento."""
    counts = Counter(train_df[label_column].tolist())
    num_samples = len(train_df)
    weights = [num_samples / (num_classes * counts.get(i, 1)) for i in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float32).to(device)


def create_model_dir(model_name: str, base_dir: Path | None = None) -> Path:
    """Crea el directorio de artefactos para un experimento."""
    base_dir = Path(base_dir) if base_dir is not None else Path("data") / "models"
    model_dir = base_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def make_json_serializable(value):
    if isinstance(value, torch.Tensor):
        return value.cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: make_json_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_serializable(v) for v in value]
    return value


def save_json(path: Path, data: dict) -> None:
    path = Path(path)
    path.write_text(json.dumps(make_json_serializable(data), indent=2), encoding="utf-8")


def save_training_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    history: dict,
    config: dict,
    class_weights: torch.Tensor,
    trainable_params: int,
    frozen_params: int,
    model_dir: Path,
    checkpoint_name: str = "checkpoint.pth",
) -> Path:
    path = model_dir / checkpoint_name
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "config": config,
            "class_weights": class_weights.cpu().tolist(),
            "trainable_params": trainable_params,
            "frozen_params": frozen_params,
        },
        path,
    )
    return path


def save_training_metadata(
    model_dir: Path,
    history: dict,
    training_meta: dict,
) -> tuple[Path, Path]:
    history_path = model_dir / "history.json"
    metadata_path = model_dir / "training_params.json"

    history_path.write_text(json.dumps(make_json_serializable(history), indent=2), encoding="utf-8")
    metadata_path.write_text(json.dumps(make_json_serializable(training_meta), indent=2), encoding="utf-8")

    return history_path, metadata_path


def save_learning_curves(history: dict, model_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    learning_curves = {
        "epochs": list(range(1, len(history["train_loss"]) + 1)),
        "train_loss": history["train_loss"],
        "val_loss": history["val_loss"],
        "train_f1": history["train_f1"],
        "val_f1": history["val_f1"],
    }

    learning_curves_path = model_dir / "learning_curves.json"
    learning_curves_path.write_text(json.dumps(make_json_serializable(learning_curves), indent=2), encoding="utf-8")

    fig = plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(learning_curves["epochs"], learning_curves["train_loss"], label="train_loss")
    plt.plot(learning_curves["epochs"], learning_curves["val_loss"], label="val_loss")
    plt.legend()
    plt.title("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(learning_curves["epochs"], learning_curves["train_f1"], label="train_f1")
    plt.plot(learning_curves["epochs"], learning_curves["val_f1"], label="val_f1")
    plt.legend()
    plt.title("F1")

    output_path = model_dir / "learning_curves.png"
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


def evaluate_and_save(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    metrics_fn,
    eval_dir: Path,
    sample_names: list[str] | None = None,
) -> dict:
    eval_dir = Path(eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)

    if sample_names is not None and len(sample_names) == len(y_true):
        predictions = pd.DataFrame(
            {
                "name": sample_names,
                "label": y_true,
                "pred": y_pred,
                "prob": y_probs,
            }
        )
        predictions.to_csv(eval_dir / "predictions.csv", index=False)

    np.save(eval_dir / "y_true.npy", y_true)
    np.save(eval_dir / "y_pred.npy", y_pred)
    np.save(eval_dir / "y_probs.npy", y_probs)

    metrics = metrics_fn(y_true, y_pred, y_probs)
    (eval_dir / "metrics.json").write_text(json.dumps(make_json_serializable(metrics), indent=2), encoding="utf-8")

    return metrics
