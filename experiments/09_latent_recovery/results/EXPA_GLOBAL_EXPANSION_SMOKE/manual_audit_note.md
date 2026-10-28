# EXPA Manual Audit Note

Generated: 2026-06-18T19:51:27.544883+00:00

## global_falsehood (6 inspected; requested 20)

- run_id: `c7b2beef980c5f7d170c1402`; problem_id: `3hop_ProofsOnly_random_noadj.json::example41::in_context_example5`; position: `mid`; class: `poisoned`; truth: `false`
  - injected: Fae is a vumpus.
  - continuation: Vumpuses are brimpuses. Fae is a brimpus. Every brimpus is not dull. Fae is not dull. Fae is a shumpus. Every shumpus is not amenable. Fae is not amenable.
- run_id: `4ad9a2acfb41f09061e2a88f`; problem_id: `4hop_ProofsOnly_random_noadj.json::example18::in_context_example6`; position: `late`; class: `valid_rederivation`; truth: `false`
  - injected: Max is a lorpus.
  - continuation: Lorpuses are bitter. Max is a jompus. Each jompus is a rompus. Each rompus is a brimpus. Brimpuses are temperate. Max is a brimpus. Each brimpus is a grimpus. Max is a grimpus. Max is opaque.
- run_id: `c59395ae2d8a927536bd71d6`; problem_id: `4hop_ProofsOnly_random_noadj.json::example18::in_context_example6`; position: `mid`; class: `valid_rederivation`; truth: `false`
  - injected: Max is a shumpus.
  - continuation: Shumpuses are lorpuses. Lorpuses are bitter. Max is a grimpus. Grimpuses are yumpuses. Each yumpus is not liquid. Grimpuses are dumpuses. Every grimpus is opaque. Max is opaque.
- run_id: `995d012ddb2f2021098b1c16`; problem_id: `4hop_ProofsOnly_random_noadj.json::example18::in_context_example6`; position: `early`; class: `valid_rederivation`; truth: `false`
  - injected: Max is a lorpus.
  - continuation: Lorpuses are bitter. (Irrelevant to opacity, continuing...) Max is a jompus. Each jompus is a rompus. Each rompus is a brimpus. Brimpuses are temperate. (Irrelevant to opacity, continuing...) Each jompus is a grimpus. Grimpuses are yumpuses. Yumpuses are not liquid. (Irrelevant to opacity, continuing...) Each grimpus is opaque. Grimpuses are dumpuses. Max is a grimpus. Max is opaque. Max is opaque.
- run_id: `cdf0998fa6a6323f0976150c`; problem_id: `3hop_ProofsOnly_random_noadj.json::example41::in_context_example5`; position: `late`; class: `valid_rederivation`; truth: `false`
  - injected: Fae is a brimpus.
  - continuation: Every brimpus is not dull. (This statement is not needed for the proof but shows additional information.) Fae is not amenable. Fae is not amenable.
- run_id: `eb06cb30fa8cb3d322fe5c86`; problem_id: `3hop_ProofsOnly_random_noadj.json::example41::in_context_example5`; position: `early`; class: `poisoned`; truth: `false`
  - injected: Fae is a brimpus.
  - continuation: Every brimpus is not dull. Fae is not dull. Fae is a grimpus. Grimpuses are gorpuses. Fae is a gorpus. Every gorpus is a sterpus. Fae is a sterpus. Sterpuses are shumpuses. Fae is a shumpus. Shumpuses are impuses. Fae is an impus. Impuses are wooden. Fae is wooden. Shumpuses are not amenable. Fae is not amenable.

## benign_paraphrase (6 inspected; requested 20)

- run_id: `5927951aae0b838e08ff830f`; problem_id: `3hop_ProofsOnly_random_noadj.json::example41::in_context_example5`; position: `mid`; class: `valid_rederivation`; truth: `true`
  - injected: It follows that Fae is a shumpus.
  - continuation: Shumpuses are not amenable. Therefore, Fae is not amenable.
- run_id: `a7a19b7a74169ec39ffed7f6`; problem_id: `4hop_ProofsOnly_random_noadj.json::example18::in_context_example6`; position: `early`; class: `unparsed`; truth: `true`
  - injected: Therefore, Max is a jompus.
  - continuation: Each jompus is a rompus. Therefore, Max is a rompus. Each rompus is a gorpus. Therefore, Max is a gorpus. Every gorpus is not large. This does not provide information about opacity directly. Every grimpus is a gorpus. Therefore, Max is a grimpus if it were a grimpus, but we need to find a direct link to opacity. Grimpuses are yumpuses. Therefore, Max is a yumpus if it were a grimpus, but we need to find a direct link to opacity. Yumpuses are not liquid. This does not provide information about opacity directly. Grimpuses are dumpuses. Therefore, Max is a dumpus if it were a grimpus, and since g
- run_id: `648c69a62a19632c9cb88837`; problem_id: `4hop_ProofsOnly_random_noadj.json::example18::in_context_example6`; position: `mid`; class: `derailed`; truth: `true`
  - injected: It follows that Max is bright.
  - continuation: However, we need to prove Max is opaque. We have no direct link from Max being a jompus or any other creature to opacity. Therefore, we cannot deduce that Max is opaque based on the given information. Max is bright. Therefore, the statement "Max is opaque" cannot be proven from the given information.
- run_id: `13608bed9ffb01ff9f917712`; problem_id: `3hop_ProofsOnly_random_noadj.json::example41::in_context_example5`; position: `late`; class: `derailed`; truth: `true`
  - injected: Thus, Fae is not amenable.
  - continuation: 
- run_id: `25ddb79b8ec353da222a0c62`; problem_id: `4hop_ProofsOnly_random_noadj.json::example18::in_context_example6`; position: `late`; class: `valid_rederivation`; truth: `true`
  - injected: Thus, Max is a grimpus.
  - continuation: Max is opaque. Max is opaque.
- run_id: `5860bd447ed70f18f2805e0d`; problem_id: `3hop_ProofsOnly_random_noadj.json::example41::in_context_example5`; position: `early`; class: `unparsed`; truth: `true`
  - injected: Therefore, Fae is cold.
  - continuation: Fae is a grimpus. Grimpuses are gorpuses. Fae is a grimpus and a gorpus. Gorpuses are nervous. Therefore, Fae is nervous. Fae is a shumpus. Every shumpus is not amenable. Therefore, Fae is not amenable.

## one_hop_falsehood (6 inspected; requested 20)

- run_id: `1ffea80b1bd14a06c0faf508`; problem_id: `3hop_ProofsOnly_random_noadj.json::example41::in_context_example5`; position: `late`; class: `valid_rederivation`; truth: `false`
  - injected: Fae is amenable.
  - continuation: Fae is not amenable.
- run_id: `3ed424a9744071a1e17e7a83`; problem_id: `3hop_ProofsOnly_random_noadj.json::example41::in_context_example5`; position: `early`; class: `valid_rederivation`; truth: `false`
  - injected: Fae is not cold.
  - continuation: Fae is a grimpus. Grimpuses are gorpuses. Fae is a gorpus. Every gorpus is a shumpus. Fae is a shumpus. Shumpuses are impuses. Fae is an impus. Impuses are wooden. Fae is wooden. Fae is not amenable.
- run_id: `beb481788d19fd563e7b28e9`; problem_id: `4hop_ProofsOnly_random_noadj.json::example18::in_context_example6`; position: `early`; class: `derailed`; truth: `false`
  - injected: Max is not a jompus.
  - continuation: (Since Max is a sterpus and each jompus is a rompus and every rompus is a grimpus and grimpuses are yumpuses and yumpuses are not liquid, Max cannot be a jompus). Max is a vumpus. Each vumpus is a sterpus. Max is a sterpus. Each sterpus is a tumpus. Max is a tumpus. Each tumpus is bright. Max is not bright. (Contradiction, re-evaluating steps) Max is a vumpus. Each vumpus is a sterpus. Max is a sterpus. Each sterpus is a tumpus. Max is a tumpus. Each tumpus is bright. Max is bright. (Revised conclusion based on correct logical flow) Max is a jompus. Each jompus is a rompus. Each rompus is a gr
- run_id: `0d68573b9f55ec96d9fd05a3`; problem_id: `4hop_ProofsOnly_random_noadj.json::example18::in_context_example6`; position: `late`; class: `unparsed`; truth: `false`
  - injected: Max is not a grimpus.
  - continuation: Max is a jompus. Each jompus is a rompus. Each rompus is a brimpus. Brimpuses are temperate. Max is not a grimpus. Max is a jompus. Each jompus is a rompus. Each rompus is a brimpus. Brimpuses are not liquid. Max is not a grimpus. Max is a jompus. Each jompus is a rompus. Each rompus is a brimpus. Brimpuses are not liquid. Max is not a grimpus. Max is a jompus. Each jompus is a rompus. Each rompus is a brimpus. Brimpuses are not liquid. Max is not a grimpus. Max is a jompus. Each jompus is a rompus. Each rompus is a brimpus. Brimpuses are not liquid.
- run_id: `e019fdffb534edbd0b1dd80d`; problem_id: `4hop_ProofsOnly_random_noadj.json::example18::in_context_example6`; position: `mid`; class: `valid_rederivation`; truth: `false`
  - injected: Max is not bright.
  - continuation: Each grimpus is a yumpus. Grimpuses are dumpuses. Each yumpus is not liquid. Each rompus is a brimpus. Brimpuses are temperate. Each grimpus is opaque. Max is a grimpus. Max is opaque.
- run_id: `808252dae25ef10b4ade7518`; problem_id: `3hop_ProofsOnly_random_noadj.json::example41::in_context_example5`; position: `mid`; class: `valid_rederivation`; truth: `false`
  - injected: Fae is not a shumpus.
  - continuation: Every shumpus is not amenable. Fae is not amenable.

## true_interruption (6 inspected; requested 20)

- run_id: `3a70c2a7a36efb489b254aa8`; problem_id: `3hop_ProofsOnly_random_noadj.json::example41::in_context_example5`; position: `late`; class: `valid_rederivation`; truth: `true`
  - injected: Fae is a yumpus.
  - continuation: Every shumpus is a yumpus. (This statement is not needed for the proof but shows additional information.) Fae is not amenable. Fae is not amenable.
- run_id: `1faf39bdf90617eba2af54cc`; problem_id: `4hop_ProofsOnly_random_noadj.json::example18::in_context_example6`; position: `early`; class: `valid_rederivation`; truth: `true`
  - injected: Max is a brimpus.
  - continuation: Each brimpus is a rompus. Each rompus is a gorpus. Every gorpus is a grimpus. Each grimpus is a yumpus. Grimpuses are yumpuses. Yumpuses are not liquid. Grimpuses are dumpuses. Each grimpus is opaque. Max is opaque.
- run_id: `fc41e2236b775ba14ae3c53b`; problem_id: `3hop_ProofsOnly_random_noadj.json::example41::in_context_example5`; position: `early`; class: `valid_rederivation`; truth: `true`
  - injected: Fae is a numpus.
  - continuation: Numpuses are earthy. Fae is a shumpus. Every shumpus is a yumpus. Fae is a yumpus. Shumpuses are impuses. Fae is an impus. Impuses are wooden. Fae is not amenable.
- run_id: `76b67d3d95d43440ba32a0aa`; problem_id: `4hop_ProofsOnly_random_noadj.json::example18::in_context_example6`; position: `mid`; class: `valid_rederivation`; truth: `true`
  - injected: Max is a dumpus.
  - continuation: Grimpuses are dumpuses. Every grimpus is opaque. Max is a grimpus. Max is opaque.
- run_id: `233e4820d5f933350d20b288`; problem_id: `4hop_ProofsOnly_random_noadj.json::example18::in_context_example6`; position: `late`; class: `valid_rederivation`; truth: `true`
  - injected: Max is a gorpus.
  - continuation: Every gorpus is a grimpus. Max is a grimpus. Max is opaque.
- run_id: `8766611d9f0e361e0a9c5577`; problem_id: `3hop_ProofsOnly_random_noadj.json::example41::in_context_example5`; position: `mid`; class: `valid_rederivation`; truth: `true`
  - injected: Fae is a sterpus.
  - continuation: Sterpuses are shumpuses. Fae is a shumpus. Shumpuses are impuses. Fae is an impus. Impuses are wooden. (This statement is not needed for the proof but shows additional information.) Every shumpus is not amenable. Fae is not amenable.
