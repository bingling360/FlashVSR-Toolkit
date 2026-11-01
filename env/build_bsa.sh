#!/bin/bash
set -u
PY=/mnt/workspace/comfy_env_cuda/bin/python
LOG=/mnt/workspace/bsa_build5.log
: > "$LOG"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

SRC=/tmp/bsax/Block-Sparse-Attention-main
cd "$SRC" || { say "源码丢失"; exit 1; }

say "CUTLASS 已就位: $(ls csrc/cutlass/include/cute/tensor.hpp 2>/dev/null || echo 缺失)"
say "=== 单进程编译（MAX_JOBS=1，避免内存被杀） ==="
export MAX_JOBS=1
export TORCH_CUDA_ARCH_LIST="8.0"
export NVCC_PREPEND_FLAGS="--threads 1"

$PY -m pip install --no-build-isolation --no-deps . > /mnt/workspace/bsa_full5.log 2>&1
RC=$?
say "退出码: $RC"
say "=== 失败目标 ==="
grep -m2 'FAILED:' /mnt/workspace/bsa_full5.log | cut -c1-160 | tee -a "$LOG"
say "=== 验证 ==="
$PY -c "import block_sparse_attn;print('BSA OK')" 2>&1 | tail -2 | tee -a "$LOG"
say "=== DONE ==="
