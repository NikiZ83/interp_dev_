import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets.embeddings_dataset import RegressionEmbeddingsDataset
from models.train_models import train_emb_model, train_probing_model
from models.probing_model import ProbingCls
import os

data_dir = "data/processed"
train_csv = os.path.join(data_dir, "train_cps.csv")
embeddings_path = "embeddings_npy"
model_save_path = "models/regression_model.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 10
batch_size = 32
input_dim = 256
learning_rate = 3e-4
source_type = "npy"
split = "train"

train_dataset = RegressionEmbeddingsDataset(
    source_path=embeddings_path,
    split=split,
    source_type=source_type,
    cps_csv=train_csv,
    collection_name="speed_embeddings"
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = ProbingCls(input_size=input_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

model = train_probing_model(
    train_loader=train_loader,
    input_dim=input_dim,
    device=device,
    num_epoch=num_epochs,
    existing_model=model
)


os.makedirs("models", exist_ok=True)
torch.save(model, model_save_path)  # Сохраняем всю модель, а не только state_dict
print(f"Модель сохранена в {model_save_path}")
