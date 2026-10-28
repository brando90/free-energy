[1] sections/02_setup.tex
  C: as the reasoning substrate.
  P: as the reasoning task.

[2] sections/02_setup.tex
  C: also match how models err naturally on this substrate.
  P: also match how models err naturally on this task.

[3] sections/03_wall.tex
  C: The dissociation is not specific to the trace substrate.
  P: The dissociation is not specific to the program-trace task.

[4] sections/09_discussion.tex
  C: Training against planted errors is the untested lever on this substrate,
  P: Training against planted errors is the untested lever on this task,

[5] sections/09_discussion.tex
  C: Our substrate is straight-line integer programs,
  P: Our task is straight-line integer programs,

[6] sections/A_appendix.tex
  C: \caption{Word-problem substrate: absorption
  P: \caption{Word-problem task: absorption

[7] sections/A_appendix.tex
  C: Each perturbed trace from the wall substrate's readable, one-operation, and copy conditions
  P: Each perturbed trace from the thirteen-model comparison's readable, one-operation, and copy conditions

[8] sections/A_appendix.tex
  C: reproduced the substrate-gap bridge row, so the on-versus-off contrast is a pure reasoning effect.
  P: reproduced its user-turn row from the thirteen-model comparison, so the on-versus-off contrast is a pure reasoning effect.

[9] sections/A_appendix.tex
  C: The protocol constants are fixed across conditions within each substrate.
  P: The protocol constants are fixed across conditions within each task format.

[10] sections/A_appendix.tex
  C: The wall substrate behind Figures 1 and 2 uses 12-line programs with the planted site at line 6,
  P: The programs behind Figures 1 and 2 have 12 lines, with the planted line at line 6,

[11] sections/02_setup.tex
  C: so the presentation does not drive the result.
  P: so the method does not drive the result.

[12] sections/03_wall.tex
  C: are measured through the user-turn presentation.
  P: are measured through the user-turn method.

[13] sections/05_opacity.tex
  C: shows absorption under bare, full, and partial presentation for the three models.
  P: shows absorption under the bare, full, and partial conditions for the three models.

[14] sections/09_discussion.tex
  C: measured through a user-turn presentation rather than prefilling,
  P: measured through the user-turn method rather than prefilling,

[15] sections/A_appendix.tex
  C: measured under the user-turn presentation, where a preface
  P: measured under the user-turn method, where a preface

[16] sections/A_appendix.tex
  C: already reported separately because it is presentation-sensitive
  P: already reported separately because it is method-sensitive

[17] sections/A_appendix.tex
  C: in line with the copy-condition inflation under this presentation recorded there.
  P: in line with the copy-condition inflation under this method recorded there.

[18] sections/A_appendix.tex
  C: \paragraph{Bridge versus prefill calibration.}
  P: \paragraph{User-turn versus prefill calibration.}

[19] sections/A_appendix.tex
  C: The bridge (user-turn) presentation and the prefill presentation were cross-calibrated on the models that support both.
  P: The user-turn method and the prefill method were cross-calibrated on the models that support both.

[20] sections/A_appendix.tex
  C: so the computed-side result does not depend on presentation.
  P: so the computed-side result does not depend on the method.

[21] sections/A_appendix.tex
  C: The pure-copy conditions inflate under the bridge method relative to prefill, so we report the copy conditions separately rather than pool the two presentations.
  P: The pure-copy conditions inflate under the user-turn method relative to prefill, so we report the copy conditions separately rather than pool the two methods.

[22] sections/A_appendix.tex
  C: showed the copy conditions measure presentation rather than recomputation.
  P: showed the copy conditions measure the method rather than recomputation.

[23] sections/04_depth.tex
  C: These arms, and the opacity and instruction arms that follow, run on
  P: These experiments, and the opacity and instruction experiments that follow, run on

[24] sections/07_reasoning.tex
  C: The counts are small but the direction is the same in both arms.
  P: The counts are small but the direction is the same for both models.

[25] sections/07_reasoning.tex
  C: so they are reported as their own arm and not pooled into
  P: so they are reported separately and not pooled into

[26] sections/07_reasoning.tex
  C: In both arms the planted error sits in the committed trace,
  P: For both models the planted error sits in the committed trace,

[27] sections/A_appendix.tex
  C: is absorbed at 0.98--1.00 in every arm on every model, and the readable error is caught in every arm
  P: is absorbed at 0.98--1.00 under every framing on every model, and the readable error is caught under every framing

[28] sections/A_appendix.tex
  C: absorbs the copy condition at 0.88--0.97 in all arms,
  P: absorbs the copy condition at 0.88--0.97 under all framings,

[29] sections/A_appendix.tex
  C: on its GSM8K random-perturbation arm with all quantities renumbered
  P: on its GSM8K random-perturbation condition with all quantities renumbered

[30] sections/A_appendix.tex
  C: restarts the solution from scratch on all 106 items in both arms rather than
  P: restarts the solution from scratch on all 106 items, original and renumbered, rather than

[31] sections/A_appendix.tex
  C: and the frontier arms are reported at that registered depth.
  P: and the Haiku 4.5 runs are reported at that registered depth.

[32] sections/A_appendix.tex
  C: The open-model depth arms run at $R = 8$ plus greedy and contribute 1350 rollouts
  P: The open-model depth runs use $R = 8$ plus greedy and contribute 1350 continuations

[33] sections/A_appendix.tex
  C: run on the same worlds with the same gold gates and grader as the depth arms,
  P: run on the same programs with the same correctness gates and grader as the depth experiments,

[34] sections/A_appendix.tex
  C: Conditions pool greedy and sampled rollouts as in the main arms;
  P: Conditions pool greedy and sampled continuations as in the main experiments;

[35] sections/A_appendix.tex
  C: \paragraph{Reasoning-arm protocol.}
  P: \paragraph{Reasoning-model protocol.}

[36] sections/A_appendix.tex
  C: come from two arms held apart from the roster.
  P: come from two runs kept separate from the thirteen-model roster.

[37] sections/A_appendix.tex
  C: The frontier arm runs GPT-5.1 through the OpenAI Responses API
  P: The GPT-5.1 run uses the OpenAI Responses API

[38] sections/A_appendix.tex
  C: The open-weight arm runs DeepSeek-R1-Distill-7B, whose gold traces are generated through its own think channel rather than by raw completion, because raw-completion gold collapses on the five-operation errors the model can only solve with that channel engaged.
  P: The open-weight run uses DeepSeek-R1-Distill-7B, whose correct base traces are generated through its own think channel rather than by raw completion, because raw completion rarely solves the five-operation programs, which the model solves only with that channel engaged.

[39] sections/A_appendix.tex
  C: Both arms are evaluated on the readable, one-operation, and five-operation conditions at the sample sizes recorded in the reasoning-arm report.
  P: Both are evaluated on the readable, one-operation, and five-operation conditions at the sample sizes recorded in the results registry.

[40] sections/A_appendix.tex
  C: was not flat across the instruction arms but drifted
  P: was not flat across the instruction conditions but drifted

[41] sections/A_appendix.tex
  C: In the depth and instruction arms (E11 and E12),
  P: In the depth and instruction experiments (E11 and E12),

[42] sections/07_reasoning.tex
  C: no re-derivation appears in 96 percent of five-operation rollouts for R1-Distill and in 85 percent for GPT-5.1 (among rollouts with
  P: no re-derivation appears in 96 percent of five-operation continuations for R1-Distill and in 85 percent for GPT-5.1 (among continuations with

[43] sections/07_reasoning.tex
  C: 2 of 9 such rollouts for R1-Distill,
  P: 2 of 9 such continuations for R1-Distill,

[44] sections/07_reasoning.tex
  C: 43 percent of its reasoning-on rollouts return an empty summary,
  P: 43 percent of its reasoning-on continuations return an empty summary,

[45] sections/09_discussion.tex
  C: Reinforcement learning on corrupted-prefix rollouts improves
  P: Reinforcement learning on corrupted-prefix samples improves

[46] sections/A_appendix.tex
  C: The fix-ladder specification (E12) called for $R = 8$ sampled rollouts per world plus one greedy rollout,
  P: The instruction-experiment specification (E12) called for $R = 8$ sampled continuations per program plus one greedy continuation,

[47] sections/A_appendix.tex
  C: so each depth-ladder condition contributes $n = 540$ rollouts over 60 programs, nine per program, and the fix ladder was likewise rerun
  P: so each depth condition contributes $n = 540$ continuations over 60 programs, nine per program, and the instruction experiment was likewise rerun

[48] sections/A_appendix.tex
  C: The depth-five conditions are attrition-limited: 504 rollouts over 56 programs
  P: The depth-five conditions are limited by how few programs are solved: 504 continuations over 56 programs

[49] sections/A_appendix.tex
  C: since rollouts of the same program are correlated;
  P: since continuations of the same program are correlated;

[50] sections/A_appendix.tex
  C: on all 360 of 360 attempted rollouts,
  P: on all 360 of 360 attempted continuations,

[51] sections/A_appendix.tex
  C: Sampling uses one greedy rollout at temperature 0 followed by $R$ sampled rollouts at temperature 0.7.
  P: Sampling uses one greedy continuation at temperature 0 followed by $R$ sampled continuations at temperature 0.7.

[52] sections/07_reasoning.tex
  C: gives a within-model contrast on the same worlds.
  P: gives a within-model contrast on the same programs.

[53] sections/A_appendix.tex
  C: Worlds are generated deterministically from an integer seed;
  P: Programs are generated deterministically from an integer seed;

[54] sections/02_setup.tex
  C: The gold-generation stage records every failed attempt
  P: The solving stage records every failed attempt

[55] sections/02_setup.tex
  C: Across the five open-weight gold runs, 1{,}775 traces contain a self-produced computed slip,
  P: Across the five open-weight solving runs, 1{,}775 traces contain a self-produced computed slip,

[56] sections/A_appendix.tex
  C: and it states $V$ (the gold gate).
  P: and it states $V$ (the correctness gate).

[57] sections/A_appendix.tex
  C: In more than 2{,}700 gold solutions,
  P: In more than 2{,}700 correct solutions,

[58] sections/A_appendix.tex
  C: The gold-generation stage also yields natural errors:
  P: The solving stage also yields natural errors:

[59] sections/A_appendix.tex
  C: Across the five open-weight gold runs, 1{,}775 traces contain one,
  P: Across the five open-weight solving runs, 1{,}775 traces contain one,

[60] sections/08_related.tex
  C: our planted contradictions sit adjacent to the site, the easy case.
  P: our planted contradictions sit adjacent to the planted line, the easy case.

[61] sections/A_appendix.tex
  C: with the planted site at line 10 (fractional position 0.5)
  P: with the planted line at line 10 (fractional position 0.5)

[62] sections/A_appendix.tex
  C: Site operands are two-digit integers drawn from 11 to 99,
  P: Operands of the planted line are two-digit integers drawn from 11 to 99,

[63] sections/A_appendix.tex
  C: and site opacity in
  P: and the planted line's opacity in

[64] main.tex
  C: under three site opacities: bare
  P: under three ways of showing the planted line: bare

[65] sections/06_fixes.tex
  C: stays at 0.99 across the ladder,
  P: stays at 0.99 across the four instruction conditions,

[66] sections/09_discussion.tex
  C: The fix ladder shows the model reads the command
  P: The instruction experiment shows the model reads the command

[67] sections/A_appendix.tex
  C: and the audited near-miss bound caps possible label mis-assignment at 0.07.
  P: and an audit of values landing within a few units of a wrong label caps possible mislabeling at 0.07.

[68] sections/A_appendix.tex
  C: a solution enters the cohort only if
  P: a solution enters the sample only if

[69] sections/A_appendix.tex
  C: so its readable condition is empty by honest starvation;
  P: so its readable condition is empty because no such solutions exist;

[70] sections/A_appendix.tex
  C: The framing is not inert as an instrument:
  P: The framing does register where it can:

[71] sections/A_appendix.tex
  C: against same-run readable anchors of 0.69, 0.67, and 0.03.
  P: against readable-error absorption of 0.69, 0.67, and 0.03 in the same runs.

