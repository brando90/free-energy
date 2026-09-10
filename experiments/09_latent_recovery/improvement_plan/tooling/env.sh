# Source this before running vllm_gen.py on skampere2.
# Keeps ALL caches off AFS (~/.cache) and pins one GPU.
export HF_HOME=/lfs/skampere2/0/eobbad/.cache/huggingface
export XDG_CACHE_HOME=/lfs/skampere2/0/eobbad/.cache
export TRITON_CACHE_DIR=/lfs/skampere2/0/eobbad/.cache/triton
export VLLM_CACHE_ROOT=/lfs/skampere2/0/eobbad/.cache/vllm
export TORCHINDUCTOR_CACHE_DIR=/lfs/skampere2/0/eobbad/.cache/torchinductor
# flashinfer ignores XDG_CACHE_HOME; cache dir = $FLASHINFER_WORKSPACE_BASE/.cache/flashinfer
export FLASHINFER_WORKSPACE_BASE=/lfs/skampere2/0/eobbad
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONDONTWRITEBYTECODE=1   # never write __pycache__ into read-only src/
export HF_HUB_OFFLINE=0
export XDG_CONFIG_HOME=/lfs/skampere2/0/eobbad/.config
export VLLM_NO_USAGE_STATS=1
