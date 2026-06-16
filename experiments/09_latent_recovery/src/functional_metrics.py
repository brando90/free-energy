"""Functional-divergence metric: rho^func at sentence horizons.

At each sentence boundary, the next-token distribution encodes "what comes next" —
the branching state of the computation, immune to lexical echo. We compare:

  JSD_pert(h) = JSD( P(next | perturbed, boundary h), P(next | gold, boundary h) )
  JSD_null(h) = mean over CORRECT null samples of the same
  rho_func(h) = JSD_pert(h) / JSD_null(h)

Boundaries: h-th sentence end AFTER the injected sentence, in each rollout's own
tokenization. Logits from saved final-layer states via the LM head (verified against
a live forward pass before use). Cohort: gold-validated instances only.
Bootstrap CIs (1000 resamples over instances) for medians.
"""
import os, json, glob
import numpy as np
import torch
from collections import defaultdict
from common import LAYERS, RESULTS, DATA, MODEL, split_sentences
from probes import sentence_token_offsets

H_MAX = 8
FINAL = str(LAYERS[-1])
FAMS = [("wrong", "perturbed"), ("distractor", "perturbed_distractor"),
        ("contradiction", "perturbed_contradiction"),
        ("paraphrase", "perturbed_paraphrase"),
        ("falsehood", "perturbed_falsehood"),
        ("negstep", "perturbed_negstep")]

def load_head():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()
    # verify hidden_states[-1] is post-final-norm: lm_head(h_last) must equal logits
    ids = tok("Every wumpus is a tumpus.", return_tensors="pt")["input_ids"].to("cuda:0")
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
        lg1 = out.logits[0, -1].float()
        lg2 = model.lm_head(out.hidden_states[-1][0, -1]).float()
    err = (lg1 - lg2).abs().max().item()
    assert err < 1e-2, f"lm_head(h_last) != logits (max err {err}) — metric invalid"
    print(f"head verification OK (max err {err:.2e})")
    return tok, model

@torch.no_grad()
def jsd(model, h_a, h_b):
    """JSD between next-token dists from two final-layer states (fp16 numpy)."""
    dev = "cuda:0"
    a = model.lm_head(torch.tensor(h_a, device=dev, dtype=torch.bfloat16)).float()
    b = model.lm_head(torch.tensor(h_b, device=dev, dtype=torch.bfloat16)).float()
    pa = torch.log_softmax(a, -1)
    pb = torch.log_softmax(b, -1)
    m = torch.logaddexp(pa, pb) - np.log(2.0)
    kl_am = (pa.exp() * (pa - m)).sum(-1)
    kl_bm = (pb.exp() * (pb - m)).sum(-1)
    return (0.5 * kl_am + 0.5 * kl_bm).cpu().numpy()

def boundary_state(hs, offs, idx):
    """final-layer state at sentence-end token of sentence idx (cont coords)."""
    if idx >= len(offs):
        return None
    pos = offs[idx] - 1
    return hs[pos] if pos < hs.shape[0] else None

def main():
    tok, model = load_head()
    data = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(DATA, "pilot.jsonl"))}
    gold = {json.loads(l)["id"]: json.loads(l)
            for l in open(os.path.join(RESULTS, "gold", "rollouts.jsonl"))}
    cohort = {json.loads(l)["id"] for l in open(os.path.join(RESULTS, "validated.jsonl"))}

    out = {"per_family": {}, "h_max": H_MAX}
    fig_data = {}
    for fam, d in FAMS:
        rpath = os.path.join(RESULTS, d, "runs.jsonl")
        if not os.path.exists(rpath):
            continue
        runs = [json.loads(l) for l in open(rpath) if "skip" not in json.loads(l)]
        rho_by_point = defaultdict(list)     # point -> list of [H_MAX] arrays (per instance)
        n_no_null = 0
        for r in runs:
            if r["id"] not in cohort:
                continue
            safe = r["id"].replace("/", "_").replace("::", "__")
            gpath = os.path.join(RESULTS, "aligned_gold", safe + ".npz")
            ppath = os.path.join(RESULTS, d, "hs", f"{safe}__{r['point']}.npz")
            if not (os.path.exists(gpath) and os.path.exists(ppath)):
                continue
            gz = np.load(gpath, allow_pickle=True)
            gmeta = json.loads(str(gz["meta"]))
            g_hs = gz[f"layer_{FINAL}"]
            steps = split_sentences(gold[r["id"]]["gen_text"])
            offs_gold = sentence_token_offsets(tok, steps)
            si = r["sent_idx"]

            pz = np.load(ppath, allow_pickle=True)
            p_hs = pz[f"layer_{FINAL}"]
            cont = split_sentences(r["continuation"])
            if not cont:
                continue
            offs_p = sentence_token_offsets(tok, cont)

            nulls = []   # (states, sentence_offsets) per correct null sample
            for npz in sorted(glob.glob(os.path.join(RESULTS, "null", "hs", f"{safe}__{r['point']}__s*.npz"))):
                nz = np.load(npz, allow_pickle=True)
                nm = json.loads(str(nz["meta"]))
                if not nm.get("correct") or "text" not in nm:
                    continue
                n_sents = split_sentences(nm["text"])
                if not n_sents:
                    continue
                nulls.append((nz[f"layer_{FINAL}"], sentence_token_offsets(tok, n_sents)))
            if len(nulls) < 2:
                n_no_null += 1
                continue

            rho_h = np.full(H_MAX, np.nan, dtype=np.float32)
            for h in range(1, H_MAX + 1):
                gi = si + h
                if gi >= len(offs_gold):
                    break
                g_pos = offs_gold[gi] - 1
                if g_pos >= g_hs.shape[0]:
                    break
                g_state = g_hs[g_pos]
                p_state = boundary_state(p_hs, offs_p, h - 1)
                if p_state is None:
                    break
                d_pert = float(jsd(model, p_state[None], g_state[None])[0])
                # null comparison at each null's OWN h-th sentence boundary
                d_nulls = []
                for nv, n_offs in nulls:
                    n_state = boundary_state(nv, n_offs, h - 1)
                    if n_state is not None:
                        d_nulls.append(float(jsd(model, n_state[None], g_state[None])[0]))
                if len(d_nulls) < 2:
                    break
                dn = float(np.mean(d_nulls))
                if dn > 1e-6:
                    rho_h[h - 1] = d_pert / dn
            rho_by_point[r["point"]].append(rho_h)

        fam_out = {}
        for p, arrs in rho_by_point.items():
            M = np.stack(arrs)
            med = np.nanmedian(M, axis=0)
            boots = []
            rng = np.random.RandomState(0)
            for _ in range(1000):
                idx = rng.randint(0, len(M), len(M))
                boots.append(np.nanmedian(M[idx], axis=0))
            B = np.stack(boots)
            fam_out[p] = {
                "n_instances": int(len(M)),
                "rho_func_median": [round(float(x), 4) if np.isfinite(x) else None for x in med],
                "ci_lo": [round(float(x), 4) if np.isfinite(x) else None for x in np.nanpercentile(B, 2.5, 0)],
                "ci_hi": [round(float(x), 4) if np.isfinite(x) else None for x in np.nanpercentile(B, 97.5, 0)],
            }
        out["per_family"][fam] = {"points": fam_out, "skipped_no_null": n_no_null}
        fig_data[fam] = rho_by_point

    with open(os.path.join(RESULTS, "functional_rho.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
    colors = {"wrong": "tab:red", "distractor": "tab:orange", "contradiction": "tab:purple"}
    for ax, p in zip(axes, ["early", "mid", "late"]):
        for fam in fig_data:
            if p not in fig_data[fam]:
                continue
            M = np.stack(fig_data[fam][p])
            med = np.nanmedian(M, axis=0)
            ax.plot(range(1, H_MAX + 1), med, marker="o", color=colors[fam],
                    label=f"{fam} (n={len(M)})")
        ax.axhline(1.0, color="k", ls="--", lw=1, alpha=0.6)
        ax.set_title(f"{p} injection")
        ax.set_xlabel("sentences after injection (h)")
    axes[0].set_ylabel(r"$\rho^{func}_h$ = JSD(pert,gold) / JSD(null,gold)")
    axes[0].legend(fontsize=9)
    fig.suptitle("Functional divergence from gold trajectory, null-calibrated "
                 r"($\rho<1$: contraction) — Qwen2.5-7B, PrOntoQA")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "rho_func.png"), dpi=150)
    print(json.dumps({f: {p: v["rho_func_median"][:4] for p, v in fo["points"].items()}
                      for f, fo in out["per_family"].items()}, indent=2))

def shuffle_control():
    """Instance-specificity control: JSD(pert, OWN gold) vs JSD(pert, OTHER gold)
    at matched sentence horizons. If similar, the metric reads generic format, not
    instance-specific computation. No null needed."""
    tok, model = load_head()
    gold = {json.loads(l)["id"]: json.loads(l)
            for l in open(os.path.join(RESULTS, "gold", "rollouts.jsonl"))}
    cohort = sorted({json.loads(l)["id"] for l in open(os.path.join(RESULTS, "validated.jsonl"))})
    runs = [json.loads(l) for l in open(os.path.join(RESULTS, "perturbed", "runs.jsonl"))
            if "skip" not in json.loads(l)]
    runs = [r for r in runs if r["id"] in set(cohort)]
    rng = np.random.RandomState(0)
    own, other = [], []
    # pre-load gold states/offsets
    gcache = {}
    for gid in cohort:
        safe = gid.replace("/", "_").replace("::", "__")
        p = os.path.join(RESULTS, "aligned_gold", safe + ".npz")
        if not os.path.exists(p):
            continue
        z = np.load(p, allow_pickle=True)
        steps = split_sentences(gold[gid]["gen_text"])
        gcache[gid] = (z[f"layer_{FINAL}"], sentence_token_offsets(tok, steps))
    ids = sorted(gcache)
    for r in runs:
        if r["id"] not in gcache:
            continue
        safe = r["id"].replace("/", "_").replace("::", "__")
        ppath = os.path.join(RESULTS, "perturbed", "hs", f"{safe}__{r['point']}.npz")
        if not os.path.exists(ppath):
            continue
        p_hs = np.load(ppath, allow_pickle=True)[f"layer_{FINAL}"]
        cont = split_sentences(r["continuation"])
        if not cont:
            continue
        offs_p = sentence_token_offsets(tok, cont)
        g_hs, offs_g = gcache[r["id"]]
        o_id = ids[rng.randint(len(ids))]
        while o_id == r["id"]:
            o_id = ids[rng.randint(len(ids))]
        o_hs, offs_o = gcache[o_id]
        si = r["sent_idx"]
        for h in range(1, 5):
            p_state = boundary_state(p_hs, offs_p, h - 1)
            gi = si + h
            if p_state is None or gi >= len(offs_g):
                break
            g_pos = offs_g[gi] - 1
            if g_pos >= g_hs.shape[0]:
                break
            own.append(float(jsd(model, p_state[None], g_hs[g_pos][None])[0]))
            oi = min(gi, len(offs_o) - 1)
            o_pos = offs_o[oi] - 1
            if o_pos < o_hs.shape[0]:
                other.append(float(jsd(model, p_state[None], o_hs[o_pos][None])[0]))
    res = {"median_jsd_own_gold": float(np.median(own)),
           "median_jsd_other_gold": float(np.median(other)),
           "n_own": len(own), "n_other": len(other),
           "ratio_other_over_own": float(np.median(other) / np.median(own))}
    with open(os.path.join(RESULTS, "shuffle_control.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "shuffle":
        shuffle_control()
    else:
        main()
