"""Shared utilities for the MCMC / Contrastive-Divergence / partition-function
experiments (questions/02_ebm_dynamic_cd_langevin, expt_v2).

Two model families live here:

* a continuous **MLP energy** `E_theta : R^d -> R` with an **SGLD/Langevin**
  sampler (used by E1, E3) — the modern "pure EBM" recipe; and
* a tiny **RBM** in numpy whose partition function `Z`, exact `log p(v)`, and
  exact maximum-likelihood gradient are all computable by enumerating the
  `2^{nv}` visible states (used by E2, E4, E5) — our ground-truth testbed.

Everything is deliberately small and dependency-light: the point is to measure
estimator bias/variance and `Z` cost cleanly, not to fit a big model.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from torch import nn


# --------------------------------------------------------------------------- #
# device / seeding
# --------------------------------------------------------------------------- #
def get_device(prefer_mps: bool = True) -> torch.device:
    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------------------------------------------------------------- #
# 2-D toy targets (continuous) — for the Langevin / score-matching EBM
# --------------------------------------------------------------------------- #
def sample_eight_gaussians(n: int, rng: np.random.Generator, std: float = 0.12,
                           radius: float = 1.6) -> np.ndarray:
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    centers = np.stack([radius * np.cos(angles), radius * np.sin(angles)], 1)
    idx = rng.integers(0, 8, size=n)
    return (centers[idx] + std * rng.standard_normal((n, 2))).astype(np.float32)


def sample_two_moons(n: int, rng: np.random.Generator, noise: float = 0.1) -> np.ndarray:
    n_a = n // 2
    n_b = n - n_a
    t_a = np.pi * rng.random(n_a)
    t_b = np.pi * rng.random(n_b)
    a = np.stack([np.cos(t_a), np.sin(t_a)], 1)
    b = np.stack([1 - np.cos(t_b), 1 - np.sin(t_b) - 0.5], 1)
    x = np.concatenate([a, b], 0)
    x += noise * rng.standard_normal(x.shape)
    # center + scale to roughly unit
    x = (x - x.mean(0)) / x.std(0)
    return x.astype(np.float32)


def sample_pinwheel(n: int, rng: np.random.Generator, n_arms: int = 5) -> np.ndarray:
    radial_std, tangential_std, rate = 0.3, 0.05, 0.25
    per = n // n_arms
    counts = [per] * n_arms
    counts[-1] += n - per * n_arms
    feats = []
    for k, c in enumerate(counts):
        r = rng.standard_normal(c) * radial_std + 1.0
        theta = rng.standard_normal(c) * tangential_std + (k * 2 * np.pi / n_arms)
        ang = theta + rate * np.exp(r)
        feats.append(np.stack([r * np.cos(ang), r * np.sin(ang)], 1))
    x = np.concatenate(feats, 0)
    return x.astype(np.float32)


DATASETS = {
    "eight_gaussians": sample_eight_gaussians,
    "two_moons": sample_two_moons,
    "pinwheel": sample_pinwheel,
}


# --------------------------------------------------------------------------- #
# continuous MLP energy + SGLD / Langevin sampler
# --------------------------------------------------------------------------- #
class MLPEnergy(nn.Module):
    """E_theta : R^d -> R, a smooth MLP scalar energy (lower = more likely)."""

    def __init__(self, dim: int = 2, hidden: int = 128, n_layers: int = 3) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last = dim
        for _ in range(n_layers - 1):
            layers += [nn.Linear(last, hidden), nn.SiLU()]
            last = hidden
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def sgld_sample(
    energy: nn.Module,
    x0: torch.Tensor,
    n_steps: int,
    step_size: float = 0.01,
    noise_scale: float | None = None,
    clamp_grad: float | None = 1.0,
    clamp_data: float | None = None,
) -> torch.Tensor:
    """Unadjusted Langevin / SGLD on a continuous EBM.

    Update:  x <- x - (step/2) * grad_x E(x) + sqrt(step) * eta,  eta ~ N(0, I).

    `noise_scale` overrides the theoretical `sqrt(step)` if given (the IGEBM
    recipe uses a small fixed noise). No Metropolis correction — this is the
    standard "short-run" sampler used to train deep EBMs.
    """
    x = x0.clone().detach().requires_grad_(True)
    noise_std = noise_scale if noise_scale is not None else float(np.sqrt(step_size))
    for _ in range(n_steps):
        e = energy(x).sum()
        (grad,) = torch.autograd.grad(e, x)
        if clamp_grad is not None:
            grad = grad.clamp(-clamp_grad, clamp_grad)
        x = x.detach() - 0.5 * step_size * grad + noise_std * torch.randn_like(x)
        if clamp_data is not None:
            x = x.clamp(-clamp_data, clamp_data)
        x = x.detach().requires_grad_(True)
    return x.detach()


# --------------------------------------------------------------------------- #
# kernel MMD (sample-quality metric)
# --------------------------------------------------------------------------- #
def mmd2_rbf(x: np.ndarray, y: np.ndarray, sigmas=(0.2, 0.5, 1.0, 2.0)) -> float:
    """Unbiased multi-bandwidth RBF MMD^2 between samples x and y."""
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)

    def _k(a, b):
        d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
        return sum(np.exp(-d2 / (2 * s * s)) for s in sigmas) / len(sigmas)

    m, n = len(x), len(y)
    kxx, kyy, kxy = _k(x, x), _k(y, y), _k(x, y)
    np.fill_diagonal(kxx, 0.0)
    np.fill_diagonal(kyy, 0.0)
    return float(kxx.sum() / (m * (m - 1)) + kyy.sum() / (n * (n - 1))
                 - 2 * kxy.mean())


# --------------------------------------------------------------------------- #
# tiny RBM in numpy — Z, exact log p(v), exact gradient all enumerable
# --------------------------------------------------------------------------- #
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def softplus(x: np.ndarray) -> np.ndarray:
    # numerically stable log(1+exp(x))
    return np.maximum(x, 0) + np.log1p(np.exp(-np.abs(x)))


def enumerate_binary(n: int) -> np.ndarray:
    """All 2^n binary row-vectors, shape (2^n, n), dtype float64."""
    idx = np.arange(2 ** n, dtype=np.int64)
    bits = ((idx[:, None] >> np.arange(n)[::-1]) & 1).astype(np.float64)
    return bits


class RBM:
    """Restricted Boltzmann Machine, binary {0,1}.

    Convention: score = NEGATIVE energy, so higher score = higher probability.

        score(v,h) = v^T W h + b^T v + c^T h,   p(v,h) ∝ exp(score)
        F(v)       = -log sum_h exp(score(v,h))
                   = -(b^T v) - sum_j softplus(c_j + (v^T W)_j)
        p(v)       ∝ exp(-F(v))

    With `nv` small we enumerate all `2^{nv}` visible states for the exact
    `log Z`, `log p(v)`, and the exact maximum-likelihood gradient.
    """

    def __init__(self, W: np.ndarray, b: np.ndarray, c: np.ndarray) -> None:
        self.W = np.asarray(W, np.float64)
        self.b = np.asarray(b, np.float64)
        self.c = np.asarray(c, np.float64)
        self.nv, self.nh = self.W.shape

    @classmethod
    def random(cls, nv: int, nh: int, rng: np.random.Generator,
               w_scale: float = 0.5, bias_scale: float = 0.2) -> "RBM":
        return cls(
            W=w_scale * rng.standard_normal((nv, nh)),
            b=bias_scale * rng.standard_normal(nv),
            c=bias_scale * rng.standard_normal(nh),
        )

    def copy(self) -> "RBM":
        return RBM(self.W.copy(), self.b.copy(), self.c.copy())

    # ---- free energy / probabilities -------------------------------------- #
    def free_energy(self, v: np.ndarray) -> np.ndarray:
        v = np.atleast_2d(v)
        return -(v @ self.b) - softplus(self.c + v @ self.W).sum(1)

    def log_Z(self) -> float:
        """Exact log partition function by enumerating all visible states."""
        v_all = enumerate_binary(self.nv)
        neg_f = -self.free_energy(v_all)
        m = neg_f.max()
        return float(m + np.log(np.exp(neg_f - m).sum()))

    def log_prob_visible(self, v: np.ndarray) -> np.ndarray:
        return -self.free_energy(v) - self.log_Z()

    def prob_visible_table(self):
        v_all = enumerate_binary(self.nv)
        logits = -self.free_energy(v_all)
        logits -= logits.max()
        p = np.exp(logits)
        p /= p.sum()
        return v_all, p

    # ---- sampling --------------------------------------------------------- #
    def sample_exact(self, n: int, rng: np.random.Generator) -> np.ndarray:
        v_all, p = self.prob_visible_table()
        idx = rng.choice(len(v_all), size=n, p=p)
        return v_all[idx].copy()

    def p_h_given_v(self, v: np.ndarray) -> np.ndarray:
        return sigmoid(self.c + v @ self.W)

    def p_v_given_h(self, h: np.ndarray) -> np.ndarray:
        return sigmoid(self.b + h @ self.W.T)

    def gibbs_step(self, v: np.ndarray, rng: np.random.Generator,
                   sample_v: bool = True):
        ph = self.p_h_given_v(v)
        h = (rng.random(ph.shape) < ph).astype(np.float64)
        pv = self.p_v_given_h(h)
        v_next = (rng.random(pv.shape) < pv).astype(np.float64) if sample_v else pv
        return v_next, h

    # ---- sufficient statistics (= d(-F)/d theta), per sample -------------- #
    def grad_stats(self, v: np.ndarray):
        """Return (dW, db, dc) of -F(v) averaged over the batch `v`.

        dW_ij = E[v_i * sigmoid(c_j+(vW)_j)], db_i = E[v_i], dc_j = E[sigmoid(.)]
        These are the sufficient statistics whose (data − model) difference is
        the maximum-likelihood gradient.
        """
        v = np.atleast_2d(v)
        ph = self.p_h_given_v(v)              # (B, nh)
        dW = (v[:, :, None] * ph[:, None, :]).mean(0)
        db = v.mean(0)
        dc = ph.mean(0)
        return dW, db, dc

    def exact_model_stats(self):
        """Negative-phase statistics under the exact model p(v)."""
        v_all, p = self.prob_visible_table()
        ph = self.p_h_given_v(v_all)                       # (S, nh)
        dW = np.einsum("s,si,sj->ij", p, v_all, ph)
        db = v_all.T @ p
        dc = ph.T @ p
        return dW, db, dc

    def exact_grad(self, data: np.ndarray):
        """Exact MLE gradient = data stats − exact model stats (flattened)."""
        pW, pb, pc = self.grad_stats(data)
        mW, mb, mc = self.exact_model_stats()
        return flatten_grad(pW - mW, pb - mb, pc - mc)


def flatten_grad(dW: np.ndarray, db: np.ndarray, dc: np.ndarray) -> np.ndarray:
    return np.concatenate([dW.ravel(), db.ravel(), dc.ravel()])


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(a @ b / (na * nb))


def integrated_autocorr_time(x: np.ndarray, max_lag: int | None = None) -> float:
    """IAT tau for a 1-D scalar chain via the initial-positive-sequence rule.

    ESS = N / tau. Larger tau = slower mixing.
    """
    x = np.asarray(x, np.float64)
    x = x - x.mean()
    n = len(x)
    if max_lag is None:
        max_lag = n // 2
    var = (x * x).mean()
    if var == 0:
        return 1.0
    tau = 1.0
    for lag in range(1, max_lag):
        ac = (x[:-lag] * x[lag:]).mean() / var
        if ac <= 0:
            break
        tau += 2 * ac
    return float(tau)
