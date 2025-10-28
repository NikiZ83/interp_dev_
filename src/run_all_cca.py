import os
import torch
from pathlib import Path
from src.cca_analysis import run_cca_analysis


os.environ["OMP_NUM_THREADS"] = "3"      # OpenMP (kaldi.fbank)
os.environ["MKL_NUM_THREADS"] = "3"      # Intel MKL (torch)
os.environ["NUMEXPR_NUM_THREADS"] = "3"  # NumExpr
os.environ["OPENBLAS_NUM_THREADS"] = "3" # OpenBLAS
torch.set_num_threads(2)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "models/voxblink2_samresnet34/voxblink2_samresnet34"
    batch_size = 2

    csv_files = [
        "dev_cps.csv",
        "test_cps.csv",
        "train_cps.csv"
    ]

    processed_dir = Path("data/processed")
    for csv_file in csv_files:
        csv_path = processed_dir / csv_file
        if csv_path.exists():
            print(f"Running CCA analysis for {csv_file}")
            run_cca_analysis(csv_path, model_name, device, batch_size)
        else:
            print(f"CSV file not found: {csv_path}")


if __name__ == "__main__":
    main()