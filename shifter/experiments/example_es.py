#!/usr/bin/env python
"""Toy experiment: shift meta-features via CTGAN + Evolution Strategies."""

from __future__ import annotations

import json, os, sys, time

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
from src.shifter import Shifter
from src.es_mfs import compute_mfe
from external.ctgan_repo.ctgan.synthesizers.ctgan_model import CTGAN
from example_california import _save_pairplot, _save_meta_plot, _print_table

# ── Config ────────────────────────────────────────────────────────────

RESULTS_ROOT = os.path.join(PROJECT_ROOT, "experiments", "example_es")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_INFER          = 2000
CTGAN_EPOCHS     = 500
CTGAN_BATCH_SIZE = 1000
CTGAN_DISC_STEPS = 5
CTGAN_LR         = 2e-4
CTGAN_VERBOSE    = True

META_FEATURES    = ["iq_range"]
SUMMARY          = "mean"
TARGET_COL       = "class"

ES_STEPS         = 300
ES_POPULATION    = 50
ES_SIGMA         = 0.05
ES_LR            = 1e-4
ES_N_SAMPLES     = 400
ES_GRAD_CLIP     = 0.5
ES_MOMENTUM      = 0.95
ES_N_Z_BATCHES   = 4

LAMBDA_Z         = 0.0
LAMBDA_X         = 0.0


# ── Helpers ───────────────────────────────────────────────────────────

def load_data(subfolder: str, filename: str = "electricity.csv") -> pd.DataFrame:
    path = os.path.join(WORKSPACE, subfolder, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if df.columns[0].startswith("Unnamed") or df.columns[0] == "":
        df = df.iloc[:, 1:]
    return df


def mfe_from_df(df: pd.DataFrame):
    df_feat = df.drop(columns=[TARGET_COL], errors="ignore")
    y = df[TARGET_COL].values if TARGET_COL in df.columns else None
    vals, names = compute_mfe(df_feat, features=META_FEATURES, summary=SUMMARY, y=y)
    return vals, names


# ── CTGAN ─────────────────────────────────────────────────────────────

def prepare_ctgan(save_dir: str) -> CTGANRepoAdapter:
    ctgan_path = os.path.join(save_dir, "trained_ctgan_electricity.pkl")

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


# ── ES param helpers ──────────────────────────────────────────────────

def flatten_params(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([p.data.reshape(-1) for p in model.parameters()])


def set_flat_params(model: torch.nn.Module, flat: torch.Tensor):
    offset = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[offset:offset + n].reshape(p.shape))
        offset += n


def set_grad_from_flat(model: torch.nn.Module, flat_grad: torch.Tensor):
    offset = 0
    for p in model.parameters():
        n = p.numel()
        g = flat_grad[offset:offset + n].reshape(p.shape)
        if p.grad is None:
            p.grad = g.clone()
        else:
            p.grad.copy_(g)
        offset += n


# ── Loss ──────────────────────────────────────────────────────────────

def evaluate_loss(shifter, adapter, target_meta, z_dim, Z_base, cond_vec) -> float:
    with torch.no_grad():
        Z_tilde = shifter(Z_base, target_meta.unsqueeze(0))
        df_tilde = adapter.generate_from_noise(Z_tilde.reshape(-1, z_dim).cpu().numpy(), cond_vec=cond_vec)
        meta_np, _ = compute_mfe(df_tilde, features=META_FEATURES, summary=SUMMARY)
        meta_t = torch.tensor(meta_np, dtype=torch.float32, device=DEVICE)
        return float(F.mse_loss(meta_t, target_meta).item())


def evaluate_loss_avg(shifter, adapter, target_meta, z_dim, Z_pool, cond_pool) -> float:
    return sum(evaluate_loss(shifter, adapter, target_meta, z_dim, Z_pool[k], cond_pool[k])
               for k in range(len(Z_pool))) / len(Z_pool)


# ── ES Training ───────────────────────────────────────────────────────

def train_shifter_es(adapter, target_meta, save_dir):
    z_dim = adapter.z_dim

    shifter = Shifter(
        z_dim=z_dim, m_dim=target_meta.shape[0], c_dim=64, hidden_dim=256, delta_scale=3.0,
    ).to(DEVICE)

    opt = torch.optim.Adam(shifter.parameters(), lr=ES_LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ES_STEPS, eta_min=1e-6)

    log, best = [], float("inf")
    grad_ema = None

    for step in range(ES_STEPS):
        theta = flatten_params(shifter)
        D = theta.numel()

        Z_pool = [torch.randn(1, ES_N_SAMPLES, z_dim, device=DEVICE) for _ in range(ES_N_Z_BATCHES)]
        cond_pool = [adapter.sample_cond_vec(ES_N_SAMPLES) for _ in range(ES_N_Z_BATCHES)]

        losses_plus, losses_minus, epsilons = [], [], []
        for _ in range(ES_POPULATION):
            eps = torch.randn(D, device=DEVICE)
            epsilons.append(eps)

            set_flat_params(shifter, theta + ES_SIGMA * eps)
            losses_plus.append(evaluate_loss_avg(shifter, adapter, target_meta, z_dim, Z_pool, cond_pool))

            set_flat_params(shifter, theta - ES_SIGMA * eps)
            losses_minus.append(evaluate_loss_avg(shifter, adapter, target_meta, z_dim, Z_pool, cond_pool))

        set_flat_params(shifter, theta)

        grad = sum((lp - lm) * eps for lp, lm, eps in zip(losses_plus, losses_minus, epsilons))
        grad /= (2.0 * ES_POPULATION * ES_SIGMA)

        grad_norm = torch.norm(grad)
        if grad_norm > ES_GRAD_CLIP:
            grad *= ES_GRAD_CLIP / grad_norm

        grad_ema = grad.clone() if grad_ema is None else ES_MOMENTUM * grad_ema + (1 - ES_MOMENTUM) * grad

        set_grad_from_flat(shifter, grad_ema)
        opt.step(); opt.zero_grad(); sched.step()

        Z_pool_eval = [torch.randn(1, ES_N_SAMPLES, z_dim, device=DEVICE) for _ in range(ES_N_Z_BATCHES)]
        cond_pool_eval = [adapter.sample_cond_vec(ES_N_SAMPLES) for _ in range(ES_N_Z_BATCHES)]
        loss = evaluate_loss_avg(shifter, adapter, target_meta, z_dim, Z_pool_eval, cond_pool_eval)
        log.append({"step": step, "loss": loss, "lr": opt.param_groups[0]["lr"]})

        if step % 20 == 0 or step == ES_STEPS - 1:
            print(f"  step {step:3d}  loss={loss:.6f}  lr={opt.param_groups[0]['lr']:.2e}")
        if loss < best:
            best = loss
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


def _save_training_plot(log, path):
    plt.figure(figsize=(10, 5))
    plt.plot([e["step"] for e in log], [e["loss"] for e in log], lw=2)
    plt.xlabel("Step"); plt.ylabel("Loss"); plt.title("ES Training"); plt.yscale("log"); plt.grid(True, alpha=.3)
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Training plot → {path}")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    os.makedirs(RESULTS_ROOT, exist_ok=True)

    print("=" * 50 + "\n[1/4] Prepare CTGAN\n" + "=" * 50)
    adapter = prepare_ctgan(RESULTS_ROOT)
    print(f"  z_dim={adapter.z_dim}")

    print("\n" + "=" * 50 + "\n[2/4] Target meta-features\n" + "=" * 50)
    df_target = load_data("target")
    meta_values, meta_names = mfe_from_df(df_target)
    print(f"  {len(meta_values)} meta-features ({META_FEATURES}, summary={SUMMARY})")
    target_meta = torch.tensor(meta_values, dtype=torch.float32, device=DEVICE)
    json.dump({"values": target_meta.cpu().tolist(), "names": meta_names},
              open(os.path.join(RESULTS_ROOT, "target_meta.json"), "w"), indent=2)

    print("\n" + "=" * 50 + "\n[3/4] Train Shifter (ES)\n" + "=" * 50)
    shifter, log = train_shifter_es(adapter, target_meta, RESULTS_ROOT)

    print("\n" + "=" * 50 + "\n[4/4] Evaluate\n" + "=" * 50)
    evaluate_shifter(shifter, adapter, target_meta, RESULTS_ROOT, log)

    print(f"\nDone in {time.time() - t0:.1f}s  →  {RESULTS_ROOT}")


if __name__ == "__main__":
    main()
