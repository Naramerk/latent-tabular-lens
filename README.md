# Latent Tabular Lens — Shifter

> A neural network for **meta-feature-conditioned latent space steering** of a pretrained CTGAN — generating tabular data with user-specified statistical properties.

> While the full study explores multiple OOD evaluation strategies, this codebase is dedicated specifically to the GAN-based latent steering approach.

---

## How it works

Standard CTGAN samples noise `Z ~ N(0, I)` and passes it through a frozen generator. **Shifter** learns to apply a small, targeted shift to `Z` so that the generated data matches a desired set of meta-features `m*`.

The CTGAN weights are **frozen throughout** — only Shifter is trained.

---

## Architecture

The shift is a residual correction:

$$\tilde{z}_i = z_i + \delta_{\text{scale}} \cdot \Delta_\theta\bigl([z_i,\ c,\ \mu_Z]\bigr)$$

- `c = MetaEncoder(m*)` — encodes target meta-features into a conditioning vector
- `μ_Z = mean(Z)` — permutation-invariant batch summary 
- `Δ_θ` — MLP predicting a per-sample shift
- `δ_scale` — shift magnitude hyperparameter 

---

## Training

**Step 1** — Train CTGAN on source data, then freeze all its weights.

**Step 2** — Train Shifter end-to-end through the frozen generator:

1. Sample fresh noise each step: `Z = torch.randn(N, z_dim)`
2. Compute shifted noise: `Z̃ = shifter(Z, m*)`
3. Generate data differentiably: `X̃ = adapter.generate_from_noise_differentiable(Z̃)`
4. Compute differentiable meta-features: `m̂ = compute_diff_mfs(X̃)`
5. Backprop only through Shifter

**Loss:**

$$\mathcal{L} = \underbrace{\text{MSE}(\hat{m},\ m^*)}_{\text{meta loss}} + \lambda_Z \cdot \underbrace{\|{\tilde{Z} - Z}\|^2}_{\text{latent reg}} + \lambda_X \cdot \underbrace{\|{\tilde{X} - X_\text{base}}\|^2}_{\text{feature reg}}$$


---

## Repository structure

```
latent-tabular-lens/
├── shifter/
│   ├── src/
│   │   ├── shifter.py              # MetaEncoder, Shifter, regularization losses
│   │   ├── differentiable_mfe.py   # Differentiable meta-features (mean, sd, cor, ...)
│   │   └── ctgan_adapter.py        # CTGANRepoAdapter: standard + differentiable generation
│   └── example/
│       ├── shifter_electricity_demo.ipynb
│       ├── shifter.pt
│       ├── trained_ctgan_iris.pkl
│       └── synthetic_shifted.csv
├── preprocessing/
│   └── tab_preprocessing.py
├── external/
│   └── ctgan_repo/                 # Custom CTGAN fork with noise-injection support
└── simple_experiment.ipynb
```
