#!/bin/bash
#SBATCH -J lif-prep -p normal -c 8 --mem=48G -t 01:00:00 -o slurm/logs/%x-%j.out
source slurm/env.sh
python3 prepare_data.py && (cd data && md5sum -c checksums.md5)
