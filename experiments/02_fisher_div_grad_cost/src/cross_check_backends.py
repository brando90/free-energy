#!/usr/bin/env python3
"""Cross-check that PyTorch and JAX compute the *same* SM loss for the same
EBM weights and inputs.

This is a correctness sanity check. We initialize one MLP-EBM in numpy,
ship the same parameters and the same x batch to both backends, and confirm
all four loss values agree:

    L_mle_like(θ; x)
    L_DSM(θ; x, noise, σ)
    L_SM_Hutch1(θ; x, v)
    L_SM_Exact(θ; x)

Usage:  python cross_check_backends.py [--in-dim 16] [--hidden 32] [--batch 8]
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from torch import nn
import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Build a shared-weights MLP-EBM via numpy
# ---------------------------------------------------------------------------


def make_weights(in_dim: int, hidden: int, depth: int, rng: np.random.Generator):
    sizes = [in_dim] + [hidden] * depth + [1]
    layers = []
    for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
        W = rng.standard_normal((fan_in, fan_out)).astype(np.float32) / np.sqrt(fan_in)
        b = np.zeros((fan_out,), dtype=np.float32)
        layers.append({"W": W, "b": b})
    return layers


# ---------------------------------------------------------------------------
# PyTorch evaluation (no autograd inside the model — manual matmul)
# ---------------------------------------------------------------------------


def pt_energy(layers_t, x):
    h = x
    for i, L in enumerate(layers_t):
        h = h @ L["W"] + L["b"]
        if i < len(layers_t) - 1:
            h = torch.nn.functional.silu(h)
    return h.squeeze(-1)


def pt_losses(layers_np, x_np, noise_np, v_np, sigma):
    f32 = torch.float32
    layers_t = [{"W": torch.tensor(L["W"], dtype=f32), "b": torch.tensor(L["b"], dtype=f32)} for L in layers_np]
    x_t = torch.tensor(x_np, dtype=f32)
    noise_t = torch.tensor(noise_np, dtype=f32)
    v_t = torch.tensor(v_np, dtype=f32)

    # MLE-like
    mle = pt_energy(layers_t, x_t).mean().item()

    # DSM
    xt = (x_t + noise_t).clone().requires_grad_(True)
    e = pt_energy(layers_t, xt).sum()
    g = torch.autograd.grad(e, xt, create_graph=False)[0]
    target = -noise_t / (sigma ** 2)
    diff = (-g) - target
    dsm = (0.5 * diff.pow(2).sum(dim=-1).mean() * (sigma ** 2)).item()

    # SM Hutchinson1
    xr = x_t.clone().requires_grad_(True)
    e = pt_energy(layers_t, xr).sum()
    g = torch.autograd.grad(e, xr, create_graph=True)[0]
    sq = 0.5 * g.pow(2).sum(dim=-1)
    gv = (g * v_t).sum()
    hvp = torch.autograd.grad(gv, xr, create_graph=False)[0]
    tr_est = (hvp * v_t).sum(dim=-1)
    smh = (sq - tr_est).mean().item()

    # SM Exact
    xe = x_t.clone().requires_grad_(True)
    e = pt_energy(layers_t, xe).sum()
    g = torch.autograd.grad(e, xe, create_graph=True)[0]
    sq = 0.5 * g.pow(2).sum(dim=-1)
    d = x_t.shape[-1]
    tr_terms = []
    for i in range(d):
        gi = g[:, i].sum()
        d2 = torch.autograd.grad(gi, xe, create_graph=True)[0][:, i]
        tr_terms.append(d2)
    tr_hess = torch.stack(tr_terms, dim=-1).sum(dim=-1)
    sme = (sq - tr_hess).mean().item()

    return {"mle_like": mle, "dsm": dsm, "sm_hutch1": smh, "sm_exact": sme}


# ---------------------------------------------------------------------------
# JAX evaluation
# ---------------------------------------------------------------------------


def jx_energy(layers_j, x):
    h = x
    for i, L in enumerate(layers_j):
        h = h @ L["W"] + L["b"]
        if i < len(layers_j) - 1:
            h = jax.nn.silu(h)
    return h.squeeze(-1)


def jx_e_single(layers_j, x_single):
    return jx_energy(layers_j, x_single[None, :])[0]


def jx_losses(layers_np, x_np, noise_np, v_np, sigma):
    layers_j = [{"W": jnp.array(L["W"]), "b": jnp.array(L["b"])} for L in layers_np]
    x_j = jnp.array(x_np)
    noise_j = jnp.array(noise_np)
    v_j = jnp.array(v_np)

    # MLE-like
    mle = float(jx_energy(layers_j, x_j).mean())

    # DSM
    def dsm_per(x, n):
        xt = x + n
        g = jax.grad(jx_e_single, argnums=1)(layers_j, xt)
        target = -n / (sigma ** 2)
        diff = (-g) - target
        return 0.5 * jnp.dot(diff, diff) * (sigma ** 2)
    dsm = float(jnp.mean(jax.vmap(dsm_per)(x_j, noise_j)))

    # SM Hutchinson1
    def smh_per(x, v):
        gf = lambda y: jax.grad(jx_e_single, argnums=1)(layers_j, y)
        g = gf(x)
        sq = 0.5 * jnp.dot(g, g)
        _, hvp = jax.jvp(gf, (x,), (v,))
        return sq - jnp.dot(hvp, v)
    smh = float(jnp.mean(jax.vmap(smh_per)(x_j, v_j)))

    # SM Exact
    def sme_per(x):
        g = jax.grad(jx_e_single, argnums=1)(layers_j, x)
        H = jax.hessian(jx_e_single, argnums=1)(layers_j, x)
        return 0.5 * jnp.dot(g, g) - jnp.trace(H)
    sme = float(jnp.mean(jax.vmap(sme_per)(x_j)))

    return {"mle_like": mle, "dsm": dsm, "sm_hutch1": smh, "sm_exact": sme}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--in-dim", type=int, default=16)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--atol", type=float, default=1e-4)
    p.add_argument("--rtol", type=float, default=1e-3)
    args = p.parse_args()

    rng = np.random.default_rng(0)
    layers = make_weights(args.in_dim, args.hidden, args.depth, rng)
    x = rng.standard_normal((args.batch, args.in_dim)).astype(np.float32)
    noise = (rng.standard_normal((args.batch, args.in_dim)).astype(np.float32) * args.sigma)
    v = rng.choice([-1.0, 1.0], size=(args.batch, args.in_dim)).astype(np.float32)

    print(f"config in_dim={args.in_dim} hidden={args.hidden} batch={args.batch}")

    pt = pt_losses(layers, x, noise, v, args.sigma)
    jx = jx_losses(layers, x, noise, v, args.sigma)

    print(f"{'loss':>10s}  {'pytorch':>14s}  {'jax':>14s}  {'abs_diff':>10s}  status")
    ok = True
    for k in pt:
        diff = abs(pt[k] - jx[k])
        tol = args.atol + args.rtol * max(abs(pt[k]), abs(jx[k]))
        status = "OK" if diff <= tol else "MISMATCH"
        if status != "OK":
            ok = False
        print(f"{k:>10s}  {pt[k]:14.6f}  {jx[k]:14.6f}  {diff:10.3e}  {status}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
