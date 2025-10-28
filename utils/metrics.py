import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.manifold import TSNE
from tqdm import tqdm

logger = logging.getLogger(__name__)

def evaluate_emb_model(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    dataset: Optional[torch.utils.data.Dataset] = None,
    layer: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None
) -> Dict[str, float]:
    """
    Evaluates an embedding model on a test dataset, computing MSE, MAE, and R2-score.

    Optionally saves predictions to a CSV file with filenames, true CPS, and predicted CPS.

    Args:
        model: The PyTorch model to evaluate.
        test_loader: DataLoader for the test dataset.
        device: Device to run the model on (e.g., 'cuda' or 'cpu').
        dataset: Dataset object containing filenames for saving predictions (optional).
        layer: Layer identifier for naming prediction columns in CSV (optional).
        save_path: Path to save predictions as a CSV file (optional).

    Returns:
        Dict containing MSE, MAE, and R2-score metrics.
    """
    model.eval()
    true_values = []
    pred_values = []
    chunk_rows = []

    with torch.no_grad():
        for i, (embeddings_batch, cps_batch) in enumerate(tqdm(test_loader, desc="Evaluating model")):
            embeddings_batch = embeddings_batch.to(device)
            outputs = model(embeddings_batch).squeeze()
            true_values.extend(cps_batch.numpy())
            pred_values.extend(outputs.cpu().numpy())

            if dataset and layer and save_path:
                batch_size = embeddings_batch.size(0)
                batch_filenames = dataset.filenames[i * test_loader.batch_size : (i + 1) * test_loader.batch_size]
                for j in range(len(cps_batch)):
                    chunk_rows.append({
                        "filename": batch_filenames[j] if j < len(batch_filenames) else "unknown",
                        "true_cps": cps_batch[j].item(),
                        f"predicted_{layer}": outputs[j].item() if outputs.dim() > 0 else outputs.item()
                    })

    metrics = {
        "mse": mean_squared_error(true_values, pred_values),
        "mae": mean_absolute_error(true_values, pred_values),
        "r2_score": r2_score(true_values, pred_values)
    }

    if chunk_rows and save_path and layer:
        save_to_csv(chunk_rows, layer, save_path)

    return metrics


def plot_tsne(embeddings: np.ndarray, true_values: np.ndarray, save_path: str) -> None:
    """
    Вычисляет t-SNE для эмбеддингов и строит plot с цветом по истинным значениям скорости речи.

    :param embeddings: Массив эмбеддингов (numpy array).
    :param true_values: Массив истинных значений скорости (для цвета).
    :param save_path: Путь для сохранения PNG.
    """
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=true_values, cmap='viridis')
    plt.colorbar(scatter, label='Speech Speed (chars/sec)')
    plt.title('t-SNE Visualization of Embeddings')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.savefig(save_path)
    plt.close()

def evaluate_probing(
    layer: str,
    y_pred: np.ndarray,
    y_true: np.ndarray
) -> Tuple[str, Dict[str, float]]:
    """
    Computes regression metrics (MSE, MAE, R2-score) for probing predictions.

    Args:
        layer: Identifier of the layer being evaluated.
        y_pred: Predicted CPS values.
        y_true: True CPS values.

    Returns:
        Tuple of layer identifier and dictionary with MSE, MAE, and R2-score.
    """
    metrics = {
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2_score": r2_score(y_true, y_pred)
    }
    return layer, metrics

def read_metrics(file_path: Union[str, Path]) -> List[Tuple[str, Dict[str, float]]]:
    """
    Reads metrics from a text file in the format saved by save_metrics.

    Args:
        file_path: Path to the metrics text file.

    Returns:
        List of tuples, each containing a layer identifier and a dictionary of metrics.
    """
    metrics_list = []
    current_layer = None
    current_metrics = {}

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_layer and current_metrics:
                    metrics_list.append((current_layer, current_metrics))
                    current_layer, current_metrics = None, {}
                continue
            if ':' not in line:
                if current_layer and current_metrics:
                    metrics_list.append((current_layer, current_metrics))
                    current_metrics = {}
                current_layer = line
            else:
                key, value = line.split(':', 1)
                current_metrics[key.strip()] = float(value.strip())

    if current_layer and current_metrics:
        metrics_list.append((current_layer, current_metrics))

    return metrics_list

def plot_metrics(metrics_list: List[Tuple[str, Dict[str, float]]], save_path: Union[str, Path]) -> None:
    """
    Creates line plots for MSE, MAE, and R2-score across layers.

    Args:
        metrics_list: List of tuples with layer identifiers and their metrics (mse, mae, r2_score).
        save_path: Path to save the plot as a PNG file.
    """
    layers = [m[0] for m in metrics_list]
    mse_values = [m[1]["mse"] for m in metrics_list]
    mae_values = [m[1]["mae"] for m in metrics_list]
    r2_values = [m[1]["r2_score"] for m in metrics_list]

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(layers, mse_values, color='#1f77b4', label="MSE", marker='o')
    plt.xticks(rotation=45, fontsize=8)
    plt.xlabel("Layers")
    plt.ylabel("Mean Squared Error")
    plt.title("MSE across Layers")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(layers, mae_values, color='#ff7f0e', label="MAE", marker='o')
    plt.xticks(rotation=45, fontsize=8)
    plt.xlabel("Layers")
    plt.ylabel("Mean Absolute Error")
    plt.title("MAE across Layers")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(layers, r2_values, color='#2ca02c', label="R2-score", marker='o')
    plt.xticks(rotation=45, fontsize=8)
    plt.xlabel("Layers")
    plt.ylabel("R2-score")
    plt.title("R2-score across Layers")
    plt.legend()

    plt.tight_layout()
    save_path = Path(save_path).with_suffix('.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Line plot saved to {save_path}")

def save_metrics(metrics_list: List[Tuple[str, Dict[str, float]]], save_path: Union[str, Path]) -> None:
    """
    Saves metrics by layer to a text file in a structured format.

    Args:
        metrics_list: List of tuples with layer identifiers and their metrics.
        save_path: Path to save the metrics text file.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'a', encoding='utf-8') as f:
        for layer, metrics in metrics_list:
            f.write(f"{layer}\n")
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")
    logger.info(f"Metrics saved to {save_path}")

def save_to_csv(chunk_rows: List[Dict], layer: str, save_path: Union[str, Path]) -> None:
    """
    Saves or updates a CSV file with model predictions for a specified layer.

    Args:
        chunk_rows: List of dictionaries with filenames, true CPS, and predicted CPS.
        layer: Layer identifier for naming prediction columns.
        save_path: Path to save or update the CSV file.
    """
    save_path = Path(save_path)
    os.makedirs(save_path.parent, exist_ok=True)
    df_new = pd.DataFrame(chunk_rows)
    predicted_col = f"predicted_{layer}"

    if save_path.exists():
        df_existing = pd.read_csv(save_path)
        if predicted_col not in df_existing.columns:
            df_existing[predicted_col] = np.nan

        for _, row in df_new.iterrows():
            filename = row["filename"]
            true_cps = row["true_cps"]
            predicted = row[predicted_col]

            if filename in df_existing["filename"].values:
                idx = df_existing[df_existing["filename"] == filename].index[0]
                if pd.isna(df_existing.at[idx, "true_cps"]):
                    df_existing.at[idx, "true_cps"] = true_cps
                if pd.isna(df_existing.at[idx, predicted_col]):
                    df_existing.at[idx, predicted_col] = predicted
            else:
                new_row = {col: np.nan for col in df_existing.columns}
                new_row["filename"] = filename
                new_row["true_cps"] = true_cps
                new_row[predicted_col] = predicted
                df_existing = pd.concat([df_existing, pd.DataFrame([new_row])], ignore_index=True)

        df_existing.to_csv(save_path, index=False)
    else:
        df_new.to_csv(save_path, index=False)
    logger.info(f"Predictions saved to {save_path}")

def plot_scatter(true_values: List[float], predicted_values: List[float], save_path: Union[str, Path]) -> None:
    """
    Creates a scatter plot comparing true and predicted CPS values.

    Args:
        true_values: List of true CPS values.
        predicted_values: List of predicted CPS values.
        save_path: Path to save the scatter plot as a PNG file.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(true_values, predicted_values, alpha=0.5, color='blue', label='Predictions')
    plt.plot([min(true_values), max(true_values)], [min(true_values), max(true_values)],
             color='red', linestyle='--', label='Ideal Line')
    plt.xlabel('True CPS')
    plt.ylabel('Predicted CPS')
    plt.title('True vs Predicted CPS')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_path = Path(save_path).with_suffix('.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Scatter plot saved to {save_path}")

def plot_error_histogram(true_values: List[float], predicted_values: List[float], save_path: Union[str, Path], bins: int = 50) -> None:
    """
    Creates a histogram of absolute prediction errors for CPS.

    Args:
        true_values: List of true CPS values.
        predicted_values: List of predicted CPS values.
        save_path: Path to save the histogram as a PNG file.
        bins: Number of bins for the histogram.
    """
    errors = np.abs(np.array(true_values) - np.array(predicted_values))
    plt.figure(figsize=(8, 6))
    plt.hist(errors, bins=bins, color='purple', alpha=0.7)
    plt.xlabel('Absolute Error (CPS)')
    plt.ylabel('Frequency')
    plt.title('Histogram of CPS Prediction Errors')
    plt.grid(True)
    plt.tight_layout()

    save_path = Path(save_path).with_suffix('.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Error histogram saved to {save_path}")

def plot_bar_metrics(metrics_list: List[Tuple[str, Dict[str, float]]], save_path: Union[str, Path]) -> None:
    """
    Creates a bar chart for model performance metrics (MSE, MAE).

    Args:
        metrics_list: List of tuples with model name and metrics dictionary (mse, mae).
        save_path: Path to save the bar chart as a PNG file.
    """
    metric_names = ['mse', 'mae']
    values = [metrics_list[0][1].get(metric, 0) for metric in metric_names]

    plt.figure(figsize=(8, 6))
    plt.bar(metric_names, values, color='green', alpha=0.7)
    plt.xlabel('Metrics')
    plt.ylabel('Value')
    plt.title('Model Performance Metrics')
    plt.grid(True, axis='y')
    plt.tight_layout()

    save_path = Path(save_path).with_suffix('.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Bar chart saved to {save_path}")