# E11/E12 open-model SYNC RECEIPT (skampere1 -> skampere2)

generated: 2026-07-28 (verified via Mac-routed md5)
synced:
  - E11_qwen7b  (COMPLETE; replaced skampere2's early world-gen-only copy)
  - E11_llama8b (NEW on skampere2)
  - E12_open    (NEW on skampere2: qwen7b + llama8b, fix-ladder arms A0-A3, .e12_open_complete)
preserved:
  - skampere2 E11_haiku (canonical, ran on skampere2 directly) -- EXCLUDED from rsync, no --delete, untouched.

rsync_rc: 0
src: skampere1:/var/tmp/eobbad/recompute_wall/free-energy/experiments/09_latent_recovery/improvement_plan/iclr_exec/recompute_wall/e11_run/results
dst: skampere2:/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery/improvement_plan/iclr_exec/recompute_wall/e11_run/results

file_count (excl E11_haiku): src=61  dst=61  (match)
md5_spotcheck (7 key files incl both summary_tables, validated_outputs, manifest, E12_open programs+marker): 7/7 MATCH (src==dst)
summary_tables parse (dest): both parse; bare cells present k0/k1/k2/k3/k5

E11 absorbed rates (dest == src):
  qwen7b : k0_bare 0.7126 (962/1350) | k1_bare 0.7844 (1059/1350)
  llama8b: k0_bare 0.5963 (805/1350) | k1_bare 0.9859 (1331/1350)

verdict: E11_SYNC_VERIFIED_OK
note: integrity confirmed by md5 computed on each node independently (routed through the
operator's Mac). During the in-node script the skampere1->skampere2 (delegated Kerberos)
ssh auth was intermittently 'Permission denied (gssapi...)' -- that only affected the
in-script verification/receipt-write, NOT the rsync (which authed and completed rc=0).
