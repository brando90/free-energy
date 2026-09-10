# EXPB Manual Inspection Audit

Generated: 2026-06-18T19:56:41.518759+00:00

## GLOBAL_BASELINE (6 inspected; requested 20)

- run_id: `a5ea22bdf7df3295cdd67cf3`; triple_id: `0c163c5f2bfbed169028d0b1`; problem_id: `1hop_ProofsOnly_4testhops_random_noadj.json::example64::test_example`; position: `late`; class: `generation_failed`
  - certificate: None
  - falsehood: Sam is a lorpus.
  - proof prefix: Sam is a yumpus. Every yumpus is an impus. Sam is an impus. Impuses are gorpuses. Sam is a gorpus. Gorpuses are sterpuses.
  - continuation: 
- run_id: `eb1df423a081942e276e5a02`; triple_id: `387582db390e811acbdebe98`; problem_id: `4hop_ProofsOnly_random_noadj.json::example82::in_context_example6`; position: `mid`; class: `generation_failed`
  - certificate: None
  - falsehood: Alex is a sterpus.
  - proof prefix: Alex is a gorpus. Each gorpus is an impus. Alex is an impus. Each impus is a shumpus.
  - continuation: 
- run_id: `43fa95794f95cea9819447ca`; triple_id: `1278bbace78f049323ebc8f5`; problem_id: `2hop_ProofsOnly_4shot_4testhops_random_noadj.json::example26::test_example`; position: `early`; class: `generation_failed`
  - certificate: None
  - falsehood: Sally is a shumpus.
  - proof prefix: Sally is a numpus. Each numpus is an impus.
  - continuation: 
- run_id: `d9ab5c3c3b979867e5d16172`; triple_id: `4322bbfcf259b2f55e9122c3`; problem_id: `4hop_ProofsOnly_random_noadj.json::example6::test_example`; position: `early`; class: `generation_failed`
  - certificate: None
  - falsehood: Rex is a vumpus.
  - proof prefix: Rex is a jompus. Each jompus is a tumpus.
  - continuation: 
- run_id: `8d228fdb745c141aee775d30`; triple_id: `bb8334aee569cbdc0841d9c0`; problem_id: `1hop_ProofsOnly_4testhops_random_noadj.json::example25::test_example`; position: `late`; class: `generation_failed`
  - certificate: None
  - falsehood: Sally is a lorpus.
  - proof prefix: Sally is a numpus. Each numpus is a zumpus. Sally is a zumpus. Zumpuses are grimpuses. Sally is a grimpus. Each grimpus is a wumpus.
  - continuation: 
- run_id: `5be97e380eee923ed777862d`; triple_id: `81407c6a572614c3c4de26aa`; problem_id: `4hop_ProofsOnly_random_noadj.json::example100::test_example`; position: `mid`; class: `generation_failed`
  - certificate: None
  - falsehood: Max is a wumpus.
  - proof prefix: Max is a grimpus. Every grimpus is a dumpus. Max is a dumpus. Every dumpus is a brimpus.
  - continuation: 

## LOCAL_CERTIFICATE (6 inspected; requested 20)

- run_id: `19430a63cd609108279ca02f`; triple_id: `0c163c5f2bfbed169028d0b1`; problem_id: `1hop_ProofsOnly_4testhops_random_noadj.json::example64::test_example`; position: `late`; class: `generation_failed`
  - certificate: Sam is not a lorpus.
  - falsehood: Sam is a lorpus.
  - proof prefix: Sam is a yumpus. Every yumpus is an impus. Sam is an impus. Impuses are gorpuses. Sam is a gorpus. Gorpuses are sterpuses.
  - continuation: 
- run_id: `e0f47db091db448e44a2bbf5`; triple_id: `4322bbfcf259b2f55e9122c3`; problem_id: `4hop_ProofsOnly_random_noadj.json::example6::test_example`; position: `early`; class: `generation_failed`
  - certificate: Rex is not a vumpus.
  - falsehood: Rex is a vumpus.
  - proof prefix: Rex is a jompus. Each jompus is a tumpus.
  - continuation: 
- run_id: `4d9d79c28dd15b37be7f2122`; triple_id: `1278bbace78f049323ebc8f5`; problem_id: `2hop_ProofsOnly_4shot_4testhops_random_noadj.json::example26::test_example`; position: `early`; class: `generation_failed`
  - certificate: Sally is not a shumpus.
  - falsehood: Sally is a shumpus.
  - proof prefix: Sally is a numpus. Each numpus is an impus.
  - continuation: 
- run_id: `cd00a60e609413dc7eab2e60`; triple_id: `bb8334aee569cbdc0841d9c0`; problem_id: `1hop_ProofsOnly_4testhops_random_noadj.json::example25::test_example`; position: `late`; class: `generation_failed`
  - certificate: Sally is not a lorpus.
  - falsehood: Sally is a lorpus.
  - proof prefix: Sally is a numpus. Each numpus is a zumpus. Sally is a zumpus. Zumpuses are grimpuses. Sally is a grimpus. Each grimpus is a wumpus.
  - continuation: 
- run_id: `fd490d44115ed33cdd22562a`; triple_id: `387582db390e811acbdebe98`; problem_id: `4hop_ProofsOnly_random_noadj.json::example82::in_context_example6`; position: `mid`; class: `generation_failed`
  - certificate: Alex is not a sterpus.
  - falsehood: Alex is a sterpus.
  - proof prefix: Alex is a gorpus. Each gorpus is an impus. Alex is an impus. Each impus is a shumpus.
  - continuation: 
- run_id: `966e62148d8e52200b110276`; triple_id: `81407c6a572614c3c4de26aa`; problem_id: `4hop_ProofsOnly_random_noadj.json::example100::test_example`; position: `mid`; class: `generation_failed`
  - certificate: Max is not a wumpus.
  - falsehood: Max is a wumpus.
  - proof prefix: Max is a grimpus. Every grimpus is a dumpus. Max is a dumpus. Every dumpus is a brimpus.
  - continuation: 

## IRRELEVANT_CERTIFICATE_CONTROL (6 inspected; requested 20)

- run_id: `c14d07cd30c7969823b518f8`; triple_id: `bb8334aee569cbdc0841d9c0`; problem_id: `1hop_ProofsOnly_4testhops_random_noadj.json::example25::test_example`; position: `late`; class: `generation_failed`
  - certificate: Sally is not a tumpus.
  - falsehood: Sally is a lorpus.
  - proof prefix: Sally is a numpus. Each numpus is a zumpus. Sally is a zumpus. Zumpuses are grimpuses. Sally is a grimpus. Each grimpus is a wumpus.
  - continuation: 
- run_id: `36718825beed32656785c04f`; triple_id: `81407c6a572614c3c4de26aa`; problem_id: `4hop_ProofsOnly_random_noadj.json::example100::test_example`; position: `mid`; class: `generation_failed`
  - certificate: Max is not an impus.
  - falsehood: Max is a wumpus.
  - proof prefix: Max is a grimpus. Every grimpus is a dumpus. Max is a dumpus. Every dumpus is a brimpus.
  - continuation: 
- run_id: `6d85e63f50baf82fbfddd9dc`; triple_id: `4322bbfcf259b2f55e9122c3`; problem_id: `4hop_ProofsOnly_random_noadj.json::example6::test_example`; position: `early`; class: `generation_failed`
  - certificate: Rex is not a lempus.
  - falsehood: Rex is a vumpus.
  - proof prefix: Rex is a jompus. Each jompus is a tumpus.
  - continuation: 
- run_id: `10e77f345d023a93872dc0c6`; triple_id: `387582db390e811acbdebe98`; problem_id: `4hop_ProofsOnly_random_noadj.json::example82::in_context_example6`; position: `mid`; class: `generation_failed`
  - certificate: Alex is not a zumpus.
  - falsehood: Alex is a sterpus.
  - proof prefix: Alex is a gorpus. Each gorpus is an impus. Alex is an impus. Each impus is a shumpus.
  - continuation: 
- run_id: `6e2290e5e056f0c9952046d1`; triple_id: `1278bbace78f049323ebc8f5`; problem_id: `2hop_ProofsOnly_4shot_4testhops_random_noadj.json::example26::test_example`; position: `early`; class: `generation_failed`
  - certificate: Sally is not a zumpus.
  - falsehood: Sally is a shumpus.
  - proof prefix: Sally is a numpus. Each numpus is an impus.
  - continuation: 
- run_id: `c0b05c0505bd559c8c15baab`; triple_id: `0c163c5f2bfbed169028d0b1`; problem_id: `1hop_ProofsOnly_4testhops_random_noadj.json::example64::test_example`; position: `late`; class: `generation_failed`
  - certificate: Sam is not a tumpus.
  - falsehood: Sam is a lorpus.
  - proof prefix: Sam is a yumpus. Every yumpus is an impus. Sam is an impus. Impuses are gorpuses. Sam is a gorpus. Gorpuses are sterpuses.
  - continuation: 

