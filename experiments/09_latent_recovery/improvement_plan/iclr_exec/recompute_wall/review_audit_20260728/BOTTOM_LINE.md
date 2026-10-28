# Bottom line — program-cluster bootstrap reanalysis (2026-07-28)

Files: cluster_boot_audit.py -> cluster_boot_results.json + cluster_boot_table.md;
cluster_boot_diffs.py -> diff_cis.json. B=2000, seed=20260728, cluster=program_id,
percentile CIs. All 45 recomputed cell counts/rates match the stored paper
summaries exactly (see "Cross-checks" in cluster_boot_table.md).

No qualitative conclusion of the paper changes under program-level clustering.
(1) The fig2 readable-vs-computed contrast is untouched: for every frontier/GPT/
Claude model and for qwen32b/llama8b the computed-minus-readable difference has a
cluster-bootstrap 95% CI bounded away from 0 (e.g. haiku +1.0000 [1.0000,1.0000];
gpt-3.5 +0.8529 [0.7647,0.9265]); the two models where it does not exclude 0
(olmo7b [-0.0588,+0.2048], qwen1p5b [-0.2563,+0.1743]) already overlapped under
Wilson and are the weak models the paper says absorb readable errors too.
Fig2 arms are greedy-only with exactly 1 rollout per program, so clustering is a
no-op there by construction (n_prog == n_roll in every cell).
(2) Clustering matters only for E11 (4 or 9 rollouts/program), and there it widens
CIs modestly without crossing any claim: haiku k1-k0 = +0.9875 [0.9708,1.0000];
llama8b +0.3896 [0.3407,0.4378]; qwen7b +0.0719 [0.0059,0.1363] — the qwen7b
depth-onset step is the only result that becomes marginal (per-cell clustered CIs
k0 [0.6674,0.7541] vs k1 [0.7348,0.8326] now overlap, though the difference CI
still excludes 0); the paper already describes qwen7b as a "gradual ramp" reaching
0.94 by depth two, and the k2 cell is far outside the k0 interval, so the claim
stands.
(3) Heterogeneity: the ceiling cells have literally zero between-program variance
(E11 haiku k1: 60/60 programs at 100%; widen llama8b/qwen32b onehop: 150/150 at
100%), which is why clustering cannot widen them — but note the percentile
bootstrap degenerates to [1.0,1.0] (or [0.0,0.0]) in those cells; the rollout
Wilson interval is the wider, more honest bound there and is what the paper reports.
Mid-rate cells are genuinely heterogeneous (e.g. E11 qwen7b k1: 6 programs at 0%,
75 at 100%, 69 intermediate), and there the cluster CI is wider than Wilson, as the
reviewer predicted — e.g. qwen7b k1 Wilson [0.7617,0.8056] vs cluster
[0.7348,0.8326] — without changing any comparison.
(4) Greedy-vs-sampled (recoverable only in E11; all fig2 arms greedy): rates agree
at the ceiling (haiku k1 greedy 1.00 = sampled 1.00); at k0, qwen7b greedy 0.8333
vs sampled 0.6975 and qwen7b k1 greedy 0.8467 vs sampled 0.7767 — pooling slightly
UNDERSTATES qwen7b absorption relative to greedy decoding, i.e. the pooling choice
is conservative with respect to the paper's wall claim.
