# Hypothesis Questions From Notebook Photos

This folder collects three hypothesis prompts extracted from notebook photos on
EBMs, Boltzmann normalization, sequence energies, transformers, and attention.

The handwriting is partly ambiguous. Each prompt therefore starts with a
transcription pass and treats the interpretation as a hypothesis to sharpen,
not as settled doctrine. Each hypothesis also carries a compact summary block:
one-sentence goal, confidence, and importance.

## Index

| ID | Folder | Coding prompt | Source image | GitHub issue | Goal | Confidence | Importance |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 00 | `00_h_boltzmann_surjection_object` | `coding_agent_prompt.md` | `assets/source_photo.jpg` | [#46](https://github.com/brando90/free-energy/issues/46) | Test whether the Boltzmann exponential `exp(-E)` is a bad hardware/inference primitive, and whether another monotone or normalized score can keep the useful ordering without paying exponential/partition-function costs. | ~75% | Very high, ~9.5/10 |
| 01 | `01_h_interpretable_energy_tags` | `coding_agent_prompt.md` | `assets/source_photo.jpg` | [#47](https://github.com/brando90/free-energy/issues/47) | Test whether reducing a rich sequence or representation to one scalar real-valued energy is too lossy for EBMs, unless the energy is structured or decomposed. | ~50% | Medium, 5/10 |
| 02 | `02_h_unsupervised_ebm_task_structure` | `coding_agent_prompt.md` | `assets/source_photo.jpg` | [#48](https://github.com/brando90/free-energy/issues/48) | Test whether transformers/attention already solve the alleged unbounded-length/full-input conditioning issue, leaving little special advantage for EBMs on this axis. | High | Very high, 9.8/10 |
