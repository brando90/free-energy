#!/bin/bash

#SBATCH --account=allocation_name
#SBATCH --partition=partition_name
### NODE/CPU/MEM/GPU  ### # can add mem gpu bind params below optionally
#SBATCH --mem-per-cpu=NG
#SBATCH --cpus-per-gpu=N
### TODO set the above mem-per-cpu and cpus-per-gpu according to your use case