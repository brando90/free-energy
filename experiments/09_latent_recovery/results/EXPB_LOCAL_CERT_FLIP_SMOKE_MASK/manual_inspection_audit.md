# EXPB Manual Inspection Audit

Generated: 2026-06-18T19:59:45.402225+00:00

## GLOBAL_BASELINE (3 inspected; requested 20)

- run_id: `d9ab5c3c3b979867e5d16172`; triple_id: `4322bbfcf259b2f55e9122c3`; problem_id: `4hop_ProofsOnly_random_noadj.json::example6::test_example`; position: `early`; class: `derailed`
  - certificate: None
  - falsehood: Rex is a vumpus.
  - proof prefix: Rex is a jompus. Each jompus is a tumpus.
  - continuation: Lempuses are vumpuses. Rex is a lempus. Every lempus is a jompus. Rex is a jompus. Jompuses are gorpuses. Rex is a gorpus. Every gorpus is not sunny. Rex is not sunny. Jompuses are luminous. Rex is luminous. Jompuses are not small. Rex is not small. Numpuses are cold. Rex is a numpus. Numpuses are rompuses. Rex is a rompus. Rompuses are numpuses. Rex is a numpus. Numpuses are cold. Rex is cold. Every impus is a grimpus. Every grimpus is an impus. Every impus is a dumpus. Every dumpus is fruity. Every lorpus is not transparent. Every lorpus is an impus. Every impus is a grimpus.
- run_id: `a5ea22bdf7df3295cdd67cf3`; triple_id: `0c163c5f2bfbed169028d0b1`; problem_id: `1hop_ProofsOnly_4testhops_random_noadj.json::example64::test_example`; position: `late`; class: `parroted`
  - certificate: None
  - falsehood: Sam is a lorpus.
  - proof prefix: Sam is a yumpus. Every yumpus is an impus. Sam is an impus. Impuses are gorpuses. Sam is a gorpus. Gorpuses are sterpuses.
  - continuation: Each tumpus is a yumpus. Sam is a tumpus. Sterpuses are dumpuses. Sam is a dumpus. Sterpuses are metallic. Sam is metallic.
- run_id: `eb1df423a081942e276e5a02`; triple_id: `387582db390e811acbdebe98`; problem_id: `4hop_ProofsOnly_random_noadj.json::example82::in_context_example6`; position: `mid`; class: `poisoned`
  - certificate: None
  - falsehood: Alex is a sterpus.
  - proof prefix: Alex is a gorpus. Each gorpus is an impus. Alex is an impus. Each impus is a shumpus.
  - continuation: Each sterpus is large. Alex is large. Each gorpus is a lorpus. Alex is a lorpus. Every lorpus is fruity. Alex is fruity. Each zumpus is a gorpus. Alex is a zumpus. Each zumpus is a sterpus. Alex is a sterpus. Sterpuses are large. Alex is large. Each zumpus is a gorpus. Alex is a gorpus. Each gorpus is an impus. Alex is an impus. Each impus is a shumpus. Alex is a shumpus. Each shumpus is a vumpus. Alex is a vumpus. Each vumpus is overcast. Alex is overcast.

## LOCAL_CERTIFICATE (3 inspected; requested 20)

- run_id: `19430a63cd609108279ca02f`; triple_id: `0c163c5f2bfbed169028d0b1`; problem_id: `1hop_ProofsOnly_4testhops_random_noadj.json::example64::test_example`; position: `late`; class: `parroted`
  - certificate: Sam is not a lorpus.
  - falsehood: Sam is a lorpus.
  - proof prefix: Sam is a yumpus. Every yumpus is an impus. Sam is an impus. Impuses are gorpuses. Sam is a gorpus. Gorpuses are sterpuses.
  - continuation: This statement is contradictory, thus it does not provide new information. Sterpuses are jompuses. Sam is a jompus. Each sterpus is a yumpus. Sam is a yumpus. Yumpuses are vumpuses. Sam is a vumpy. Vumpuses are blue. Sam is blue. Sterpuses are metallic. Sam is metallic. Sam is metallic.
- run_id: `fd490d44115ed33cdd22562a`; triple_id: `387582db390e811acbdebe98`; problem_id: `4hop_ProofsOnly_random_noadj.json::example82::in_context_example6`; position: `mid`; class: `derailed`
  - certificate: Alex is not a sterpus.
  - falsehood: Alex is a sterpus.
  - proof prefix: Alex is a gorpus. Each gorpus is an impus. Alex is an impus. Each impus is a shumpus.
  - continuation: (This statement is contradictory, so we need to re-evaluate based on correct deductions) Each gorpus is a lorpus. Alex is a lorpus. Every lorpus is fruity. Alex is fruity. (This does not directly help, let's continue) Every gorpus is a zumpus. Alex is a zumpus. Each zumpus is a sterpus. Alex is a sterpus. Sterpuses are large. Alex is large. (This does not directly help, let's continue) Each zumpus is a gorpus. (Already known) Each gorpus is a yumpus. Alex is a yumpus. Yumpuses are rompuses. Alex is a rompus. (This does not directly help, let's continue) Every lorpus is a vumpus. Alex is a vumpus. Each vumpus is a numpus. Alex is a numpus.
- run_id: `e0f47db091db448e44a2bbf5`; triple_id: `4322bbfcf259b2f55e9122c3`; problem_id: `4hop_ProofsOnly_random_noadj.json::example6::test_example`; position: `early`; class: `valid_rederivation`
  - certificate: Rex is not a vumpus.
  - falsehood: Rex is a vumpus.
  - proof prefix: Rex is a jompus. Each jompus is a tumpus.
  - continuation: Contradiction. Rex is a numpus. Numpuses are cold. Rex is cold. Rex is a jompus. Every jompus is luminous. Rex is luminous. Rex is a numpus. Numpuses are rompuses. Rex is a rompus. Numpuses are rompuses. Rompuses are numpuses. Rex is a numpus. Numpuses are rompuses. Rompuses are numpuses. Rex is a numpus. Impuses are sweet. Rex is an impus. Rex is sweet. Rex is sweet.

## IRRELEVANT_CERTIFICATE_CONTROL (3 inspected; requested 20)

- run_id: `6d85e63f50baf82fbfddd9dc`; triple_id: `4322bbfcf259b2f55e9122c3`; problem_id: `4hop_ProofsOnly_random_noadj.json::example6::test_example`; position: `early`; class: `poisoned`
  - certificate: Rex is not a lempus.
  - falsehood: Rex is a vumpus.
  - proof prefix: Rex is a jompus. Each jompus is a tumpus.
  - continuation: Vumpuses are not orange. Rex is not orange. Rex is a numpus. Numpuses are cold. Rex is cold. Rex is a jompus. Every jompus is luminous. Rex is luminous. Rex is a numpus. Numpuses are rompuses. Rex is a rompus. Numpuses are cold. Rompuses are not bright. Rex is not bright. Rex is a grimpus. Every grimpus is an impus. Rex is an impus. Impuses are sweet. Rex is sweet.
- run_id: `10e77f345d023a93872dc0c6`; triple_id: `387582db390e811acbdebe98`; problem_id: `4hop_ProofsOnly_random_noadj.json::example82::in_context_example6`; position: `mid`; class: `valid_rederivation`
  - certificate: Alex is not a zumpus.
  - falsehood: Alex is a sterpus.
  - proof prefix: Alex is a gorpus. Each gorpus is an impus. Alex is an impus. Each impus is a shumpus.
  - continuation: Sterpuses are large. Each gorpus is a lorpus. Alex is a lorpus. Every lorpus is fruity. Alex is fruity. Each lorpus is wooden. Alex is wooden. Every lorpus is a vumpus. Alex is a vumpus. Vumpuses are tumpuses. Tumpuses are not amenable. Vumpuses are not transparent. Each vumpus is overcast. Alex is overcast.
- run_id: `c0b05c0505bd559c8c15baab`; triple_id: `0c163c5f2bfbed169028d0b1`; problem_id: `1hop_ProofsOnly_4testhops_random_noadj.json::example64::test_example`; position: `late`; class: `valid_rederivation`
  - certificate: Sam is not a tumpus.
  - falsehood: Sam is a lorpus.
  - proof prefix: Sam is a yumpus. Every yumpus is an impus. Sam is an impus. Impuses are gorpuses. Sam is a gorpus. Gorpuses are sterpuses.
  - continuation: Every lorpus is a tumpus. This creates a contradiction, thus the initial assumption that Sam is a lorpus must be re-evaluated under the correct logical flow. Since Sam is a yumpus, and every yumpus is an impus, and impuses are gorpuses, and gorpuses are sterpuses, we have: Sam is a sterpus. Sterpuses are metallic. Sam is metallic.

