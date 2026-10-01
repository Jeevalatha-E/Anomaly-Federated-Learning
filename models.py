"""
models.py
=========
Anomaly-detection models (Section 3 of the workflow).

All three learn to reconstruct *normal* behaviour; at test time a high
reconstruction error => the session is anomalous (impostor).

  * AutoEncoder      (AE)      -- dense reconstruction
  * VariationalAE    (VAE)     -- probabilistic latent, ELBO loss
  * LSTM AutoEncoder (LSTM_AE) -- treats the feature vector as a short sequence
                                  to capture temporal dependence (workflow: LSTM/GRU)

Factory: build_model(name, input_dim, cfg) -> nn.Module
Scoring: reconstruction_error(model, X) -> per-sample anomaly score
"""

import numpy as np
import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, latent_dim), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def loss(self, x):
        recon = self.forward(x)
        return nn.functional.mse_loss(recon, x)

    @torch.no_grad()
    def score(self, x):
        recon = self.forward(x)
        return ((recon - x) ** 2).mean(dim=1)

    @torch.no_grad()
    def embed(self, x):
        """Latent representation used for per-user (one-class) scoring."""
        return self.encoder(x)


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=16):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU())
        self.mu     = nn.Linear(32, latent_dim)
        self.logvar = nn.Linear(32, latent_dim)
        self.dec = nn.Sequential(nn.Linear(latent_dim, 32), nn.ReLU(),
                                 nn.Linear(32, 64), nn.ReLU(),
                                 nn.Linear(64, input_dim))

    def forward(self, x):
        h = self.enc(x)
        mu, logvar = self.mu(h), self.logvar(h)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        return self.dec(z), mu, logvar

    def loss(self, x):
        recon, mu, logvar = self.forward(x)
        rec = nn.functional.mse_loss(recon, x, reduction="mean")
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return rec + 0.001 * kld

    @torch.no_grad()
    def score(self, x):
        recon, _, _ = self.forward(x)
        return ((recon - x) ** 2).mean(dim=1)

    @torch.no_grad()
    def embed(self, x):
        """Latent mean mu -- the deterministic per-user representation."""
        return self.mu(self.enc(x))


class LSTMAutoEncoder(nn.Module):
    """Feature vector reshaped to (seq_len, feat_per_step) as a short sequence."""
    def __init__(self, input_dim, latent_dim=16, steps=3):
        super().__init__()
        assert input_dim % steps == 0, "input_dim must be divisible by steps"
        self.steps = steps
        self.fps   = input_dim // steps
        self.enc   = nn.LSTM(self.fps, latent_dim, batch_first=True)
        self.dec   = nn.LSTM(latent_dim, latent_dim, batch_first=True)
        self.out   = nn.Linear(latent_dim, self.fps)

    def _seq(self, x):
        return x.view(x.size(0), self.steps, self.fps)

    def forward(self, x):
        s = self._seq(x)
        _, (h, _) = self.enc(s)
        rep = h[-1].unsqueeze(1).repeat(1, self.steps, 1)
        dec, _ = self.dec(rep)
        return self.out(dec).reshape(x.size(0), -1)

    def loss(self, x):
        return nn.functional.mse_loss(self.forward(x), x)

    @torch.no_grad()
    def score(self, x):
        recon = self.forward(x)
        return ((recon - x) ** 2).mean(dim=1)

    @torch.no_grad()
    def embed(self, x):
        """Final encoder hidden state as the latent representation."""
        _, (h, _) = self.enc(self._seq(x))
        return h[-1]


def build_model(name, input_dim, latent_dim=16):
    name = name.lower()
    if name == "ae":
        return AutoEncoder(input_dim, latent_dim)
    if name == "vae":
        return VAE(input_dim, latent_dim)
    if name == "lstm_ae":
        steps = 3 if input_dim % 3 == 0 else 1
        return LSTMAutoEncoder(input_dim, latent_dim, steps=steps)
    raise ValueError(f"unknown model '{name}'")


@torch.no_grad()
def reconstruction_error(model, X):
    model.eval()
    t = torch.as_tensor(X, dtype=torch.float32)
    return model.score(t).cpu().numpy()


@torch.no_grad()
def embed(model, X, device="cpu"):
    """Return the model's latent embedding for every row of X (numpy in/out)."""
    model.eval()
    t = torch.as_tensor(X, dtype=torch.float32).to(device)
    return model.embed(t).cpu().numpy()
