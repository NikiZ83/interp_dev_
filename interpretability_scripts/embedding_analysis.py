import argparse
import os
from utils import (
    get_loaders,
    save_emb_metrics,
    save_visualization,
    evaluate_emb_model
)
from models import train_emb_model, EmbeddingModel

import torch
import torch.nn as nn
import torch.optim as optim


def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embeddings_source",
        type=str,
        choices=["npy", "chromadb"],
        required=True,
        help="Source for embeddings: npy or chromadb"
    )
    parser.add_argument(
        "--source_path",
        type=str,
        default="./embeddings",
        help="Path to npy file or to chromadb collection folder"
    )
    parser.add_argument(
        "--eval_path",
        type=str,
        default="./scores/gender.txt",
        help="Save path for evaluation results file (txt)"
    )
    parser.add_argument(
        "--visual_path",
        type=str,
        default="./result/gender.png",
        help="Save path for embeddings visualisation"
    )
    args = parser.parse_args()

    if not os.path.exists(args.source_path):
        raise FileNotFoundError(f"Folder {args.source_path} does not exists.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader, test_dataset, input_dim = get_loaders(
        args.source_path, args.embeddings_source
    )
    model = EmbeddingModel(input_dim, 2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    train_emb_model(model, train_loader, optimizer,
                    criterion, num_epoch=300, device=device)

    metrics = evaluate_emb_model(model, test_loader, device)
    save_emb_metrics(metrics, args.eval_path)
    save_visualization(
        model, test_dataset.audio_data.numpy(),
        test_dataset.labels.numpy(), args.visual_path, device=device
    )


if __name__ == '__main__':
    main()
