from .probing_model import ProbingCls

import torch
import torch.optim as optim
from tqdm import tqdm


def train_emb_model(
        model,
        train_loader,
        optimizer,
        criterion,
        num_epoch,
        device
):
    """
    Train a model on a train dataset for regression on speech rate (CPS)
    """
    for epoch in tqdm(range(num_epoch), desc="Training Progress"):
        model.train()

        for embeddings_batch, labels_batch in tqdm(
                train_loader, desc=f"Epoch {epoch + 1}/{num_epoch}"
        ):
            embeddings_batch = embeddings_batch.to(device)

            labels_batch = labels_batch.float()
            _, outputs = model(embeddings_batch)
            loss = criterion(outputs.squeeze(1), labels_batch.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def train_probing_model(
        train_loader,
        input_dim,
        device,
        num_epoch=3,
        existing_model=None
):
    """
    Train a probing model on a train dataset for regression on speech rate (CPS).
    Assumes embeddings from WeSpeaker (as per extract_embeddings.md) and CPS labels from preprocess.md.
    """
    model = existing_model or ProbingCls(input_dim).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    model.train()
    for epoch in tqdm(range(num_epoch), desc="Training Progress"):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device).float()

            optimizer.zero_grad()
            outputs = model(X_batch).squeeze(1)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

    return model