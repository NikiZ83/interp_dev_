import argparse
import json
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


def compute_cca(X, Y):
    if X.shape[1] < 1 or Y.shape[1] < 1:
        return 0.0
    cca = CCA(n_components=1, max_iter=500)
    try:
        cca.fit(X, Y)
        X_c, Y_c = cca.transform(X, Y)
        corr = np.corrcoef(X_c[:, 0], Y_c[:, 0])[0, 1]
    except Exception as e:
        print(f"CCA error: {e}")
        corr = 0.0
    return corr


def extract_activations_for_layer(activator, audio_paths, layer, device, batch_size=16):
    activations = []
    num_samples = len(audio_paths)
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_paths = audio_paths[start_idx:end_idx]

        batch_feats = []
        for path in batch_paths:
            feats = extract_features(path).squeeze(0)  # (T, 80)
            batch_feats.append(feats)

        max_len = max(f.shape[0] for f in batch_feats)
        padded_feats = torch.zeros(len(batch_feats), max_len, 80, device=device)
        for i, f in enumerate(batch_feats):
            padded_feats[i, :f.shape[0], :] = f

        # Удалить эту строку: padded_feats = padded_feats.unsqueeze(1)

        acts_dict, _ = activator(padded_feats, target_layer=layer, from_activation=False, identity_file=None)
        act = acts_dict[layer].detach().cpu().numpy()  # (B, C, H, W) or (B, D)
        act_flat = act.reshape(act.shape[0], -1)  # (B, features)
        activations.append(act_flat)

    return np.concatenate(activations, axis=0)

def run_cca_analysis(csv_path: Path, model_name: str, device: str, batch_size: int = 16,
                     output_dir: Path = Path("results_1/cca")):
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if 'train' in csv_path.stem:
        df = df.sample(n=2000, random_state=42)
    audio_paths = df['audio_path'].tolist()
    cps_values = df['cps'].values  # (N,)

    model = wespeaker.load_model(model_name)
    model.set_device(device)
    activator = GetActivations(model)
    layers = get_layers(model)

    cca_scores = {}
    for layer in tqdm(layers, desc="Processing layers and computing CCA"):
        acts = extract_activations_for_layer(activator, audio_paths, layer, device, batch_size)
        X = acts  # (N, D)
        Y = cps_values[:, np.newaxis]  # (N, 1)
        score = compute_cca(X, Y)
        cca_scores[layer] = score
        del acts, X

    scores_df = pd.DataFrame.from_dict(cca_scores, orient='index', columns=['cca_corr'])
    scores_path = output_dir / f"cca_{csv_path.stem}.csv"
    scores_df.to_csv(scores_path)

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(cca_scores)), list(cca_scores.values()), align='center')
    plt.xticks(range(len(cca_scores)), list(cca_scores.keys()), rotation=90)
    plt.ylabel("CCA correlation with CPS")
    plt.title(f"CCA per layer - {csv_path.stem}")
    plt.tight_layout()
    plot_path = output_dir / f"cca_{csv_path.stem}.png"
    plt.savefig(plot_path)
    plt.close()

    print(f"Results saved: {scores_path}, {plot_path}")
    return scores_df, plot_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CCA analysis on LibriSpeech subsets")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to CPS CSV file")
    parser.add_argument("--model_name", type=str, default="voxblink2_samresnet34", help="WeSpeaker model name")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for activation extraction")
    args = parser.parse_args()

    run_cca_analysis(Path(args.csv_path), args.model_name, args.device, args.batch_size)