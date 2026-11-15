#!/bin/bash
#SBATCH -J fineweb-prep -p normal -c 16 --mem=32G -t 04:00:00 -o slurm/logs/%x-%j.out
# General-text data (README section 0), step 1b: GPT-2 token streams for FineWeb-Edu (prepare_fineweb.py), after
# slurm/fineweb_download.sh has filled data/fineweb/raw. From the repository root:  sbatch slurm/prepare_fineweb.sbatch
source slurm/env.sh
python3 prepare_fineweb.py && ls -la data/fineweb && (cd data/fineweb && md5sum gpt2.json valid.bin)
