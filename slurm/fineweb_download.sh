#!/bin/bash
# General-text data (README section 0), step 1a: download the 10-billion-token sample of FineWeb-Edu (Hugging Face
# HuggingFaceFW/fineweb-edu, sample/10BT: 14 Parquet files, 28.5 GB) into data/fineweb/raw. Needs internet: run it
# where PG-19 was downloaded (slurm/notes.txt). Files already there are skipped, so it can be rerun after a failure.
# Then step 1b on a CPU node:  sbatch slurm/prepare_fineweb.sbatch   (the raw files can be deleted afterwards).
set -e
cd "$(dirname "$0")/.."
source slurm/env.sh
python3 -c "import pyarrow" 2>/dev/null || pip install -q pyarrow
mkdir -p data/fineweb/raw
files=$(curl -sSf https://huggingface.co/api/datasets/HuggingFaceFW/fineweb-edu/tree/main/sample/10BT |
        python3 -c "import json, sys; print('\n'.join(x['path'] for x in json.load(sys.stdin) if x['path'].endswith('.parquet')))")
for f in $files; do
  out=data/fineweb/raw/$(basename "$f")
  [ -s "$out" ] && continue
  curl -sSfL --retry 5 -o "$out.part" "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/$f" && mv "$out.part" "$out"
  echo "$out $(du -h "$out" | cut -f1)"
done
echo "$(ls data/fineweb/raw/*.parquet | wc -l) files, $(du -sh data/fineweb/raw | cut -f1)"
