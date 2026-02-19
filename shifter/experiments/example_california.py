#!/usr/bin/env python
"""Toy experiment: shift meta-features via CTGAN + differentiable backprop."""

from __future__ import annotations

import json, os, sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

WORKSPACE = os.path.dirname(PROJECT_ROOT)
CTGAN_LIB = os.path.join(WORKSPACE, "latent-tabular-lens")
if os.path.exists(CTGAN_LIB):
    sys.path.insert(0, CTGAN_LIB)

from src.ctgan_adapter import CTGANRepoAdapter
from src.shifter import Shifter, latent_reg, feature_space_reg
from src.diff_mfs import compute_diff_mfs
from external.ctgan_repo.ctgan.synthesizers.ctgan_model import CTGAN

# ── Config ────────────────────────────────────────────────────────────

RESULTS_ROOT = os.path.join(PROJECT_ROOT, "experiments", "example_california")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_INFER          = 2000
CTGAN_EPOCHS     = 500
CTGAN_BATCH_SIZE = 1000
CTGAN_DISC_STEPS = 5
CTGAN_LR         = 2e-4
CTGAN_VERBOSE    = True

META_FEATURES    = ["mut_inf"]
SUMMARY          = None

TRAIN_STEPS      = 1000
TRAIN_LR         = 3e-4
TRAIN_N_SAMPLES  = 400
TRAIN_GRAD_CLIP  = 0.5
TRAIN_ACCUM      = 10

LAMBDA_Z         = 0.0
LAMBDA_X         = 0.000005
TARGET_COL       = "price_above_median"
DELTA_SCALE_VEC  = 0.5
DELTA_SCALE_SCL  = 0.1


# ── Helpers ───────────────────────────────────────────────────────────

def load_data(subfolder: str, filename: str = "california_Longitude.csv") -> pd.DataFrame:
    path = os.path.join(WORKSPACE, subfolder, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if df.columns[0].startswith("Unnamed") or df.columns[0] == "":
        df = df.iloc[:, 1:]
    return df


def get_feature_col_indices(adapter: CTGANRepoAdapter, df: pd.DataFrame):
    indices, names = [], []
    for idx, ci in enumerate(adapter.m._transformer._column_transform_info_list):
        if ci.column_name != TARGET_COL and ci.column_name in df.columns and df[ci.column_name].dtype.kind in "iuf":
            indices.append(idx)
            names.append(ci.column_name)
    return torch.tensor(indices, dtype=torch.long, device=DEVICE), names


def mfe_from_df(df: pd.DataFrame):
    X = torch.tensor(
        df.drop(columns=[TARGET_COL], errors="ignore")
          .select_dtypes(include=[np.number]).values,
        dtype=torch.float32, device=DEVICE,
    )
    with torch.no_grad():
        vals, names = compute_diff_mfs(X, features=META_FEATURES, summary=SUMMARY)
    return vals.cpu().numpy().flatten(), names


# ── CTGAN ─────────────────────────────────────────────────────────────

def prepare_ctgan(save_dir: str) -> CTGANRepoAdapter:
    ctgan_path = os.path.join(save_dir, f"trained_ctgan_california_Longitude.pkl")

    if os.path.exists(ctgan_path):
        print(f"  Loading cached CTGAN from {ctgan_path}")
        ctgan = torch.load(ctgan_path, map_location=DEVICE, weights_only=False)
    else:
        df_train = load_data("source")
        print(f"  Training CTGAN on {len(df_train)} rows …")
        cat_cols = [c for c in df_train.columns
                    if df_train[c].dtype == "object" or df_train[c].dtype.name == "category" or c == TARGET_COL]
        ctgan = CTGAN(
            epochs=CTGAN_EPOCHS, discriminator_steps=CTGAN_DISC_STEPS,
            batch_size=CTGAN_BATCH_SIZE, generator_lr=CTGAN_LR, discriminator_lr=CTGAN_LR,
            pac=1, cuda=False, verbose=CTGAN_VERBOSE,
        )
        ctgan.set_device(DEVICE)
        ctgan.fit(train_data=df_train, discrete_columns=cat_cols)
        torch.save(ctgan, ctgan_path)
        print(f"  CTGAN saved → {ctgan_path}")

    ctgan.set_device(DEVICE)
    adapter = CTGANRepoAdapter(ctgan)
    
    adapter.m._generator.eval()
    for p in adapter.m._generator.parameters():
        p.requires_grad_(False)
    
    return adapter


# ── Training ──────────────────────────────────────────────────────────

def compute_loss(shifter, adapter, target_meta, Z_base, cond_vec, X_source, feat_idx):
    z_dim = adapter.z_dim
    Z_tilde = shifter(Z_base.unsqueeze(0), target_meta.unsqueeze(0))
    X_tilde = adapter.generate_from_noise_differentiable(Z_tilde.reshape(-1, z_dim), cond_vec=cond_vec)
    X_tilde = X_tilde[:, feat_idx]
    meta_tilde, _ = compute_diff_mfs(X_tilde, features=META_FEATURES, summary=SUMMARY)

    if SUMMARY is None and meta_tilde.shape[0] > 1:
        rel = (meta_tilde - target_meta) / (target_meta.abs() + 1e-6)
        loss = (rel ** 2).mean()
    else:
        loss = F.mse_loss(meta_tilde, target_meta)

    if LAMBDA_Z > 0:
        loss = loss + LAMBDA_Z * latent_reg(Z_tilde, Z_base.unsqueeze(0))
    if LAMBDA_X > 0:
        n = X_tilde.shape[0]
        loss = loss + LAMBDA_X * feature_space_reg(X_tilde, X_source[torch.randperm(len(X_source), device=DEVICE)[:n]])
    return loss


def train_shifter(adapter, target_meta, save_dir):
    z_dim = adapter.z_dim
    df_source = load_data("source")
    feat_idx, col_names = get_feature_col_indices(adapter, df_source)
    print(f"  {len(feat_idx)} numeric feature columns (excl. '{TARGET_COL}')")

    X_source = torch.tensor(df_source[col_names].values, dtype=torch.float32, device=DEVICE)

    shifter = Shifter(
        z_dim=z_dim, m_dim=target_meta.shape[0], c_dim=64, hidden_dim=256,
        delta_scale=DELTA_SCALE_VEC if SUMMARY is None else DELTA_SCALE_SCL,
    ).to(DEVICE)

    opt = torch.optim.Adam(shifter.parameters(), lr=TRAIN_LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=TRAIN_STEPS, eta_min=1e-5)

    log, best = [], float("inf")

    for step in range(TRAIN_STEPS):
        opt.zero_grad()
        total_loss = 0.0
        
        for k in range(TRAIN_ACCUM):
            Z_base = torch.randn(TRAIN_N_SAMPLES, z_dim, device=DEVICE)
            cond_vec = adapter.sample_cond_vec(TRAIN_N_SAMPLES)
            
            loss = compute_loss(shifter, adapter, target_meta, Z_base, cond_vec, X_source, feat_idx)
            (loss / TRAIN_ACCUM).backward()
            total_loss += loss.item()
        
        avg_loss = total_loss / TRAIN_ACCUM
        torch.nn.utils.clip_grad_norm_(shifter.parameters(), TRAIN_GRAD_CLIP)
        opt.step(); sched.step()

        log.append({"step": step, "loss": avg_loss, "lr": opt.param_groups[0]["lr"]})
        if step % 20 == 0 or step == TRAIN_STEPS - 1:
            print(f"  step {step:3d}  loss={avg_loss:.6f}  lr={opt.param_groups[0]['lr']:.2e}")
        if avg_loss < best:
            best = avg_loss
            torch.save(shifter.state_dict(), os.path.join(save_dir, "shifter.pt"))

    json.dump(log, open(os.path.join(save_dir, "train_log.json"), "w"), indent=2)
    print(f"  Best loss: {best:.6f}")
    return shifter, log


# ── Evaluation ────────────────────────────────────────────────────────

def evaluate_shifter(shifter, adapter, target_meta, save_dir, train_log):
    shifter.eval()
    df_source = load_data("source")
    meta_source, meta_names = mfe_from_df(df_source)

    df_base = adapter.m.sample(n=min(N_INFER, len(df_source)))
    meta_base, _ = mfe_from_df(df_base)

    with torch.no_grad():
        torch.manual_seed(42)
        Z = torch.randn(1, N_INFER, shifter.z_dim, device=DEVICE)
        Z_tilde = shifter(Z, target_meta.unsqueeze(0))
        df_shifted = adapter.generate_from_noise(Z_tilde.reshape(-1, shifter.z_dim).cpu().numpy())
    meta_shifted, _ = mfe_from_df(df_shifted)

    target_np = target_meta.cpu().numpy().flatten()
    _print_table(meta_names, meta_source, meta_base, meta_shifted, target_np)

    df_base.to_csv(os.path.join(save_dir, "generated_base.csv"), index=False)
    df_shifted.to_csv(os.path.join(save_dir, "generated_shifted.csv"), index=False)

    _save_pairplot(df_base, df_shifted, os.path.join(save_dir, "pairplot.png"))
    _save_training_plot(train_log, os.path.join(save_dir, "training_plot.png"))
    _save_meta_plot(meta_source, target_np, meta_base, meta_shifted, meta_names,
                    os.path.join(save_dir, "meta_features_vectors.png"))


def _print_table(names, source, base, shifted, target):
    print(f"\n  {'Feature':<25s} {'Source':>12s} {'Base':>12s} {'Diff%':>8s} {'Shifted':>12s} {'Target':>12s}")
    print("  " + "-" * 95)
    for i, n in enumerate(names):
        d = abs((base[i] - source[i]) / source[i] * 100) if source[i] else float("inf")
        print(f"  {n:<25s} {source[i]:>12.4f} {base[i]:>12.4f} {d:>7.1f}% {shifted[i]:>12.4f} {target[i]:>12.4f}")


# ── Plots ─────────────────────────────────────────────────────────────

def _save_pairplot(df_base, df_shifted, path):
    cols = df_base.select_dtypes(include=[np.number]).columns.tolist()[:6]
    combined = pd.concat([
        df_base[cols].head(2000).assign(source="Base"),
        df_shifted[cols].head(2000).assign(source="Shifted"),
    ], ignore_index=True)
    sns.pairplot(combined, hue="source", diag_kind="kde",
                 plot_kws={"alpha": .5, "s": 8}, palette={"Base": "blue", "Shifted": "red"})
    plt.suptitle("Base vs Shifted", y=1.02, fontsize=14)
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Pairplot → {path}")


def _save_training_plot(log, path):
    plt.figure(figsize=(10, 5))
    plt.plot([e["step"] for e in log], [e["loss"] for e in log], lw=2)
    plt.xlabel("Step"); plt.ylabel("Loss"); plt.title("Training"); plt.yscale("log"); plt.grid(True, alpha=.3)
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Training plot → {path}")


def _save_meta_plot(src, tgt, base, shifted, names, path):
    n = len(names)
    x, w = np.arange(n), 0.2
    fig, ax = plt.subplots(figsize=(16, 6))
    for offset, data, label, color in [
        (-1.5, src, "Source", "blue"), (-0.5, tgt, "Target", "green"),
        (0.5, base, "Base Synthetic", "orange"), (1.5, shifted, "Shifted Synthetic", "red"),
    ]:
        ax.bar(x + offset * w, data, w, label=label, alpha=0.8, color=color)
    ax.set_xlabel("Meta-feature Component"); ax.set_ylabel("Value")
    ax.set_title("Meta-features Comparison", fontweight="bold")
    ax.legend(); ax.grid(True, alpha=.3, axis="y"); ax.set_xticks(x)
    labels = [n.split(".")[-1] if "." in n else n for n in names] if n <= 30 else [f"col{i}" for i in range(n)]
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Meta-features plot → {path}")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_ROOT, exist_ok=True)

    print("=" * 50 + "\n[1/4] Prepare CTGAN\n" + "=" * 50)
    adapter = prepare_ctgan(RESULTS_ROOT)
    print(f"  z_dim={adapter.z_dim}")

    print("\n" + "=" * 50 + "\n[2/4] Target meta-features\n" + "=" * 50)
    df_target = load_data("target")
    meta_values, meta_names = mfe_from_df(df_target)
    print(f"  {len(meta_values)} components ({META_FEATURES}, summary={SUMMARY})")
    target_meta = torch.tensor(meta_values, dtype=torch.float32, device=DEVICE)
    json.dump({"values": target_meta.cpu().tolist(), "names": meta_names},
              open(os.path.join(RESULTS_ROOT, "target_meta.json"), "w"), indent=2)

    print("\n" + "=" * 50 + "\n[3/4] Train Shifter\n" + "=" * 50)
    shifter, log = train_shifter(adapter, target_meta, RESULTS_ROOT)

    print("\n" + "=" * 50 + "\n[4/4] Evaluate\n" + "=" * 50)
    evaluate_shifter(shifter, adapter, target_meta, RESULTS_ROOT, log)


if __name__ == "__main__":
    main()
