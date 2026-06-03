"""Sanity check: actually train an EBM via score matching.

Continuous analog of `experiments/01_toy_ebm_training/run_toy_ebm.py` (which
uses a discrete binary support and MLE-style positive/negative-phase
updates). Here we generate a Gaussian-mixture target on R^D, and train the
TinyEBM with one of:

  exact_sm | hutch_sm | sliced_sm | dsm

so that we can confirm the score-matching gradient (the thing we profiled
in profile_sm_*.py) actually drives the model toward the data distribution.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from ebm_models import TinyEBM  # noqa: E402
from profile_sm_pytorch import (  # noqa: E402
    loss_dsm,
    loss_exact_sm,
    loss_hutch_sm,
    loss_sliced_sm,
)


LOSS_TABLE = {
    "exact_sm": loss_exact_sm,
    "hutch_sm": loss_hutch_sm,
    "sliced_sm": loss_sliced_sm,
    "dsm": loss_dsm,
}


def sample_gmm(n: int, D: int, sep: float = 3.0, seed: int = 0) -> torch.Tensor:
    """Two-component isotropic Gaussian mixture in R^D, equal weights."""
    g = torch.Generator().manual_seed(seed)
    components = torch.randint(0, 2, (n,), generator=g)
    mu = torch.zeros(2, D)
    mu[0, 0] = -sep
    mu[1, 0] = +sep
    x = mu[components] + torch.randn(n, D, generator=g)
    return x


def true_score(x: torch.Tensor, sep: float = 3.0) -> torch.Tensor:
    """Closed-form score nabla_x log p_data for the 2-component GMM above.

    log p(x) = log( 0.5 N(x; -sep e_1, I) + 0.5 N(x; +sep e_1, I) )
    score = sum_k r_k(x) (mu_k - x)
    where r_k is the posterior weight on component k.
    """
    D = x.shape[-1]
    mu = torch.zeros(2, D, device=x.device, dtype=x.dtype)
    mu[0, 0] = -sep
    mu[1, 0] = +sep
    # log unnormalized component densities
    log_w = -0.5 * ((x.unsqueeze(1) - mu) ** 2).sum(dim=-1)  # (B, 2)
    r = torch.softmax(log_w, dim=-1)  # (B, 2)
    # E_k[mu_k] - x
    return (r.unsqueeze(-1) * (mu.unsqueeze(0) - x.unsqueeze(1))).sum(dim=1)


def score_match_error(model, x, sep) -> float:
    """E_x ||score_model(x) - score_true(x)||."""
    x = x.detach().requires_grad_(True)
    e = model(x).sum()
    s_model = -torch.autograd.grad(e, x, create_graph=False)[0]
    s_true = true_score(x.detach(), sep=sep)
    err = (s_model - s_true).norm(dim=-1).mean().item()
    return err


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--loss",
        choices=list(LOSS_TABLE),
        default="hutch_sm",
        help="Which SM estimator to train with.",
    )
    p.add_argument("--dim", type=int, default=8)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--out_json",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "results" / "train_sm_toy.json",
    )
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    sep = 3.0
    data = sample_gmm(args.batch * 4, args.dim, sep=sep, seed=args.seed).to(device)
    model = TinyEBM(args.dim, args.hidden, num_layers=3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = LOSS_TABLE[args.loss]
    history = []
    t0 = time.perf_counter()
    for step in range(args.steps):
        idx = torch.randint(0, data.shape[0], (args.batch,), device=device)
        x = data[idx]
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model, x)
        loss.backward()
        opt.step()
        if step % max(1, args.steps // 20) == 0 or step == args.steps - 1:
            with torch.enable_grad():
                # use a fresh, independent eval batch to avoid in-place reuse
                eval_x = sample_gmm(512, args.dim, sep=sep, seed=args.seed + 1).to(device)
                err = score_match_error(model, eval_x, sep)
            history.append(
                {"step": step, "loss": float(loss.item()), "score_err": err}
            )
            print(
                f"step {step:5d}  loss={loss.item():+8.4f}  "
                f"||score_model - score_true||={err:.4f}",
                flush=True,
            )
    elapsed = time.perf_counter() - t0

    payload = {
        "loss": args.loss,
        "dim": args.dim,
        "hidden": args.hidden,
        "batch": args.batch,
        "steps": args.steps,
        "lr": args.lr,
        "device": str(device),
        "elapsed_s": elapsed,
        "final": history[-1],
        "history": history,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    out = args.out_json
    if out.exists():
        out = out.with_name(f"{out.stem}_{args.loss}.json")
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}  ({elapsed:.1f} s total)")


if __name__ == "__main__":
    main()
