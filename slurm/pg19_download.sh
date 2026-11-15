#!/bin/bash
# Long-text track (README 5.22), step 1a: download PG-19 (the Parquet mirror emozilla/pg19 on Hugging Face, 25 files,
# about 7 GB) into data/pg19/raw, and install pyarrow into the venv if it is missing. Run on the LOGIN node, which has
# internet (compute nodes may not):  bash slurm/pg19_download.sh
# Then step 1b on a CPU node:  sbatch slurm/prepare_pg19.sbatch
set -e
cd "$(dirname "$0")/.."
source slurm/env.sh
python3 -c "import pyarrow" 2>/dev/null || pip install -q pyarrow
mkdir -p data/pg19/raw
files=$(curl -sSf https://huggingface.co/api/datasets/emozilla/pg19/tree/main/data |
        python3 -c "import json, sys; print('\n'.join(x['path'] for x in json.load(sys.stdin) if x['path'].endswith('.parquet')))")
for f in $files; do
  out=data/pg19/raw/$(basename "$f")
  [ -s "$out" ] && continue
  curl -sSfL --retry 5 -o "$out.part" "https://huggingface.co/datasets/emozilla/pg19/resolve/main/$f" && mv "$out.part" "$out"
  echo "$out $(du -h "$out" | cut -f1)"
done
echo "$(ls data/pg19/raw/*.parquet | wc -l) files, $(du -sh data/pg19/raw | cut -f1)"
