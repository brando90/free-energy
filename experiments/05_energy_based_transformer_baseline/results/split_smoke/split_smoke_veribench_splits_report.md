# VeriBench EBT Split Report

Status: `warn`

## Train / Val / Test Summary

| split | phase | MRR | all gold top | mean gold rank | mean margin |
| --- | --- | ---: | ---: | ---: | ---: |
| train | before | 0.236 | False | 4.33 | -0.0341 |
| train | after | 0.288 | False | 3.67 | -0.0434 |
| val | before | 0.196 | False | 5.25 | -0.0232 |
| val | after | 0.258 | False | 4.00 | -0.3336 |
| test | before | 0.237 | False | 4.50 | -0.0034 |
| test | after | 0.279 | False | 3.75 | -0.2385 |

## Test Tasks After Training

| task | candidates | gold rank | top candidate | best negative - gold energy |
| --- | ---: | ---: | --- | ---: |
| compilers_set_synthetic/MyLexerSpanCover | 6 | 5 | wrong_task_gold__compilers_set_synthetic_MyWriteEnableGate | -0.5771 |
| compilers_set_synthetic/MyPeepholeAddZero | 6 | 4 | wrong_task_gold__compilers_set_synthetic_MyWriteEnableGate | -0.3768 |
| compilers_set_synthetic/MyWriteEnableGate | 6 | 3 | gold | 0.0000 |
| concurrent_set_real/MyRaftVoteGate | 6 | 3 | gold | 0.0000 |

Lower energy is better. Ties with negatives count against the gold rank.
