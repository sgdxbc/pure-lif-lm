#!/bin/bash
#SBATCH -J pg19-prep -p normal -c 16 --mem=64G -t 04:00:00 -o slurm/logs/%x-%j.out
# Long-text track (README 5.22), step 1b: tokenizer + token streams for PG-19 (prepare_pg19.py), after
# slurm/pg19_download.sh has filled data/pg19/raw. From the repository root:  sbatch slurm/prepare_pg19.sbatch
source slurm/env.sh
python3 prepare_pg19.py && ls -la data/pg19 && (cd data/pg19 && md5sum tok8192.json valid.bin test.bin)
