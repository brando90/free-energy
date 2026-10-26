# Codex E2/E4 cross-check verdict
n_seeds used: 800.
Config: nv=14, nh=6, n_data=4000, batch=128, data_seed=0, eval_seed=1, w_scale=1.5.
E2 CD bias_rel by k [1,2,5,10,20,50]: 0.468009, 0.25657, 0.0602378, 0.00780521, 0.00325227, 0.00321768.
E2 CD cosine by k [1,2,5,10,20,50]: 0.921312, 0.979656, 0.998763, 0.999974, 0.999995, 0.999995.
CONFIRM (a): yes; bias_rel is monotone decreasing and cosine at k=50 is 0.999995.
PCD bias_rel=0.00426831; CD-1 bias_rel=0.468009.
CONFIRM (b): yes; PCD bias is lower than CD-1 bias.
E4 MSE by schedule: last=0.123362, uniform=0.093331, geom_late=0.0653963, zipf_late=0.0623277, early=0.807191.
Best E4 schedule by gradient MSE: zipf_late.
CONFIRM (c): no; geom_late, zipf_late beat both uniform and early, while last does not.
