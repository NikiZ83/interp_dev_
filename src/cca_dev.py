import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from sklearn.cross_decomposition import CCA
import wespeaker

from utils.layers import GetActivations, get_layers
from utils.extract_features import extract_features


# os.environ["OMP_NUM_THREADS"] = "4"
# os.environ["MKL_NUM_THREADS"] = "4"
# os.environ["NUMEXPR_NUM_THREADS"] = "4"
# os.environ["OPENBLAS_NUM_THREADS"] = "4"
# torch.set_num_threads(4)


def compute_cca(X: np.ndarray, Y: np.ndarray) -> float:
    if X.shape[0] < 2:
        return 0.0
    cca = CCA(n_components=1, max_iter=500)
    try:
        cca.fit(X, Y)
        X_c, Y_c = cca.transform(X, Y)
        return np.corrcoef(X_c[:, 0], Y_c[:, 0])[0, 1]
    except Exception as e:
        print(f"CCA error: {e}")
        return 0.0

def extract_activations_for_layer(
    activator,
    audio_paths,
    layer,
    device,
    max_frames=400,
) -> np.ndarray:
    acts = []
    N = len(audio_paths)

    for i, path in enumerate(audio_paths):
        feats = extract_features(path).squeeze(0).to(device)

        T = feats.shape[0]
        if T > max_frames:
            feats = feats[:max_frames]
        else:
            pad = torch.zeros(max_frames - T, 80, device=device)
            feats = torch.cat([feats, pad], dim=0)

        feats = feats.unsqueeze(0)  # (1, 400, 80)

        act_dict, _ = activator(
            feats, target_layer=layer, from_activation=False, identity_file=None
        )
        act = act_dict[layer].detach().cpu().numpy()

        if len(act.shape) == 4:  # (1, C, H, W)
            act = np.mean(act, axis=(2, 3))  # (1, C)

        act_flat = act.reshape(1, -1)  # (1, D)
        acts.append(act_flat)

        del feats, act_dict, act, act_flat
    return np.concatenate(acts, axis=0)  # (N, D)


def run_cca_analysis(
    csv_path: Path,
    model_name: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    n_samples: int = 200,
    max_frames: int = 400,
    output_dir: Path = Path("results_1/cca"),
):
    if n_samples < 2:
        raise ValueError("n_samples must be >= 2 for CCA")

    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    if len(df) > n_samples:
        df = df.sample(n=min(n_samples, len(df)), random_state=42)

    audio_paths = df["audio_path"].tolist()
    cps = df["cps"].values.astype(np.float32)  # cps

    print(f"Using {len(audio_paths)} samples from {csv_path.name}")

    model = wespeaker.load_model(model_name)
    model.set_device(device)
    activator = GetActivations(model)
    layers = get_layers(model)

    scores = {}

    for layer in tqdm(layers, desc="Layers"):
        try:
            X = extract_activations_for_layer(activator, audio_paths, layer, device, max_frames)
            Y = cps[:, np.newaxis]
            score = compute_cca(X, Y)
            scores[layer] = score
            print(f"{layer}: {score:.4f}")
            del X, Y
        except Exception as e:
            print(f"Failed {layer}: {e}")
            scores[layer] = 0.0

    # Сохранение результатов
    scores_df = pd.DataFrame.from_dict(scores, orient="index", columns=["cca_corr"])
    csv_out = output_dir / f"cca_{csv_path.stem}.csv"
    scores_df.to_csv(csv_out)

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(scores)), list(scores.values()))
    plt.xticks(range(len(scores)), list(scores.keys()), rotation=90, fontsize=8)
    plt.ylabel("CCA correlation")
    plt.title(f"CCA – {csv_path.stem} ({len(audio_paths)} samples)")
    plt.tight_layout()
    png_out = output_dir / f"cca_{csv_path.stem}.png"
    plt.savefig(png_out, dpi=150)
    plt.close()

    print(f"Saved: {csv_out}, {png_out}")
    return scores_df, png_out


if __name__ == "__main__":
    dev_csv = Path("data/processed/dev_cps.csv")
    if not dev_csv.exists():
        print(f"File not found: {dev_csv}")
    else:
        run_cca_analysis(
            csv_path=dev_csv,
            model_name="models/voxblink2_samresnet100_ft/voxblink2_samresnet100_ft",
            device="cuda" if torch.cuda.is_available() else "cpu",
            n_samples=500,  # ≥2
            max_frames=400,
        )