import os
import chromadb
import numpy as np
import pandas as pd
import torch
from .base_dataset import BaseDataset


class RegressionEmbeddingsDataset(BaseDataset):
    def __init__(
            self,
            source_path,
            split,
            source_type,
            cps_csv,
            collection_name="speed_embeddings"
    ):
        """
        Dataset for loading embeddings and CPS values for regression tasks.

        Args:
            source_path (str): Path to the directory with embeddings (.npy or ChromaDB).
            split (str): Dataset split ('train', 'test', 'dev').
            source_type (str): Type of embedding storage ('npy' or 'chromadb').
            cps_csv (str): Path to the CSV file with CPS values (e.g., 'data/processed/train_cps.csv').
            collection_name (str): Name of the ChromaDB collection (default: 'speed_embeddings').
        """
        super().__init__()
        self.source_path = source_path
        self.split = split
        self.source_type = source_type
        self.cps_csv = cps_csv
        self.collection_name = collection_name

        # Check if CPS CSV file exists
        if not os.path.exists(self.cps_csv):
            raise FileNotFoundError(f"CPS CSV file {self.cps_csv} does not exist.")

        self.prepare_data()

    def prepare_data(self):
        """
        Loads and prepares embeddings, CPS values, and filenames based on the source type.
        """
        if self.source_type == "npy":
            audio_data, cps_values, filenames = self.get_npy_embeddings()
        elif self.source_type == "chromadb":
            audio_data, cps_values, filenames = self.get_chroma_embeddings()
        else:
            raise ValueError(f"Invalid source type: {self.source_type}. Choose 'npy' or 'chromadb'.")

        self.audio_data = torch.tensor(audio_data, dtype=torch.float32)
        self.labels = torch.tensor(cps_values, dtype=torch.float32)
        self.filenames = filenames  # Store filenames as an attribute

    def get_npy_embeddings(self):
        """
        Reads embeddings from a .npy file and matches them with CPS values from a CSV file.

        Returns:
            tuple: (embeddings, cps_values, filenames) where embeddings is a numpy array of embeddings,
                   cps_values is a list of CPS values, and filenames is a list of file paths.
        """
        # Load embeddings
        npy_path = os.path.join(self.source_path, "numpy_embs.npy")
        if not os.path.exists(npy_path):
            raise FileNotFoundError(f"NumPy file {npy_path} does not exist.")

        source = np.load(npy_path, allow_pickle=True).item()  # Load as dictionary
        data = source.get(self.split, [])
        if not data:
            raise ValueError(f"No data found for split '{self.split}' in {npy_path}.")

        embeddings = np.array([item['embedding'] for item in data], dtype=np.float32)
        filenames = [item['file_path'] for item in data]

        # Load CPS values from CSV
        cps_data = pd.read_csv(self.cps_csv)
        path_to_cps = {row['audio_path']: row['cps'] for _, row in cps_data.iterrows()}
        cps_values = [path_to_cps.get(fname, 0.0) for fname in filenames]  # Fallback to 0.0

        return embeddings, cps_values, filenames

    def get_chroma_embeddings(self):
        """
        Reads embeddings from ChromaDB and matches them with CPS values from a CSV file.

        Returns:
            tuple: (embeddings, cps_values, filenames) where embeddings is a numpy array of embeddings,
                   cps_values is a list of CPS values, and filenames is a list of file paths.
        """
        # Load embeddings from ChromaDB
        client = chromadb.PersistentClient(path=self.source_path)
        collection = client.get_or_create_collection(name=self.collection_name)
        results = collection.get(where={"split": self.split}, include=["embeddings", "metadatas"])

        if not results['embeddings']:
            raise ValueError(f"No embeddings found for split '{self.split}' in ChromaDB collection '{self.collection_name}'.")

        embeddings = np.array(results['embeddings'], dtype=np.float32)
        filenames = [item['file_path'] for item in results['metadatas']]

        # Load CPS values from CSV
        cps_data = pd.read_csv(self.cps_csv)
        path_to_cps = {row['audio_path']: row['cps'] for _, row in cps_data.iterrows()}
        cps_values = [path_to_cps.get(fname, 0.0) for fname in filenames]  # Fallback to 0.0

        return embeddings, cps_values, filenames

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.audio_data)

    def __getitem__(self, idx):
        """
        Returns a single sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (embedding, cps_value) where embedding is a tensor of the audio embedding,
                   and cps_value is the corresponding CPS value.
        """
        return self.audio_data[idx], self.labels[idx]