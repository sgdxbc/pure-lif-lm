# Source from every job: venv + all caches in home (the NFS home is the large storage; login-node /tmp has a 10 MB quota).
source ~/venv-lif/bin/activate
export XDG_CACHE_HOME=~/.cache PIP_CACHE_DIR=~/.cache/pip TORCH_HOME=~/.cache/torch HF_HOME=~/.cache/huggingface
export TRITON_CACHE_DIR=~/.cache/triton TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions
export TMPDIR=${SLURM_TMPDIR:-/tmp}
cd ~/pure-lif-lm
