import torch
from torch.utils.data import DataLoader
from datasets.embeddings_dataset import RegressionEmbeddingsDataset
from utils.metrics import evaluate_emb_model, save_metrics, plot_metrics, plot_scatter, plot_error_histogram, \
    plot_bar_metrics, save_to_csv, plot_tsne
import logging
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = Path("models/regression_model.pt")
    test_csv = Path("data/processed/test_cps.csv")
    embeddings_path = Path("embeddings_npy")
    source_type = "npy"
    collection_name = "speed_embeddings"
    results_dir = Path("results_1")
    metrics_path = results_dir / "metrics.txt"
    scatter_plot_path = results_dir / "scatter_plot.png"
    error_hist_path = results_dir / "error_histogram.png"
    bar_plot_path = results_dir / "bar_metrics.png"
    predictions_csv_path = results_dir / "predictions.csv"
    metrics_plot_path = results_dir / "metrics_plot.png"
    tsne_plot_path = results_dir / "tsne_plot.png"

    for path in [model_path, test_csv, embeddings_path]:
        if not path.exists():
            logger.error(f"File or directory not found: {path}")
            return

    try:
        logger.info(f"Loading test dataset from {test_csv}...")
        test_dataset = RegressionEmbeddingsDataset(
            source_path=str(embeddings_path),
            split="test",
            source_type=source_type,
            cps_csv=str(test_csv),
            collection_name=collection_name
        )
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
        logger.info(f"Test dataset loaded, size: {len(test_dataset)}")

        logger.info(f"Loading model from {model_path}...")
        model = torch.load(model_path, weights_only=False, map_location=device)
        model.to(device)
        model.eval()

        logger.info("Evaluating model...")
        metrics = evaluate_emb_model(
            model=model,
            test_loader=test_loader,
            device=device,
            dataset=test_dataset,
            layer="model",
            save_path=str(predictions_csv_path)
        )
        logger.info(f"Model metrics: {metrics}")

        metrics_list = [("model", metrics)]
        save_metrics(metrics_list, str(metrics_path))
        logger.info(f"Metrics saved to {metrics_path}")

        logger.info("Generating metrics plot...")
        plot_metrics(metrics_list, str(metrics_plot_path))
        logger.info(f"Metrics plot saved to {metrics_plot_path}")

        logger.info("Collecting predictions for visualization...")
        true_values = []
        predicted_values = []
        all_embeddings = []
        with torch.no_grad():
            for embeddings_batch, cps_batch in test_loader:
                embeddings_batch = embeddings_batch.to(device)
                outputs = model(embeddings_batch).squeeze().cpu().numpy()
                true_values.extend(cps_batch.numpy())
                predicted_values.extend(outputs)
                all_embeddings.extend(embeddings_batch.cpu().numpy())

        all_embeddings = np.array(all_embeddings)
        true_values_array = np.array(true_values)

        logger.info("Generating t-SNE visualization...")
        plot_tsne(all_embeddings, true_values_array, str(tsne_plot_path))
        logger.info(f"t-SNE plot saved to {tsne_plot_path}")

        logger.info("Generating visualizations...")
        plot_scatter(true_values, predicted_values, str(scatter_plot_path))
        logger.info(f"Scatter plot saved to {scatter_plot_path}")

        plot_error_histogram(true_values, predicted_values, str(error_hist_path))
        logger.info(f"Error histogram saved to {error_hist_path}")

        plot_bar_metrics(metrics_list, str(bar_plot_path))
        logger.info(f"Bar chart saved to {bar_plot_path}")

    except Exception as e:
        logger.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()