#!/bin/bash
# ============================================================
# 环境准备：依赖检查 + 修复已知问题
#
# 用法： bash setup_env.sh [ComfyUI根目录]
# ============================================================
set -u

COMFY="${1:-/mnt/workspace/ComfyUI}"
PY="${COMFY%/ComfyUI}/comfy_env_cuda/bin/python"
[ -x "$PY" ] || PY=python

say(){ echo "[$(date +%H:%M:%S)] $*"; }

say "=== 1. 检查 Python 与关键包 ==="
$PY - <<'EOF'
import importlib, sys
print("  Python", sys.version.split()[0])
for m in ("torch", "numpy", "PIL", "einops", "imageio"):
    try:
        mod = importlib.import_module(m)
        v = getattr(mod, "__version__", "?")
        print(f"  ✓ {m} {v}")
    except Exception:
        print(f"  ✗ {m} 缺失")
try:
    import torch
    print("  CUDA", torch.version.cuda, "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "无 GPU")
except Exception:
    pass
EOF

say "=== 2. 检查 block_sparse_attn（FlashVSR 必需）==="
$PY -c "import torch, block_sparse_attn; print('  ✓ 已安装')" 2>/dev/null \
  || say "  ✗ 未安装 —— 请先运行 env/build_bsa.sh（约 70 分钟）"

say "=== 3. 修复 transformer_engine 导致的节点加载失败 ==="
# 症状：SeedVR2 节点报 "cudnn shared object not found"
# 原因：系统路径里的 transformer_engine 缺少 cudnn 依赖，被 peft/diffusers 触发
SITE=$($PY -c "import site;print(site.getsitepackages()[0])" 2>/dev/null)
if [ -n "${SITE:-}" ] && [ -d "$SITE/transformer_engine" ]; then
  mv "$SITE/transformer_engine" "$SITE/transformer_engine.disabled" && \
    say "  已屏蔽 $SITE/transformer_engine"
else
  say "  无需处理（未找到或已屏蔽）"
fi

say "=== 4. 检查 ffmpeg（流式节点需要）==="
command -v ffmpeg >/dev/null && say "  ✓ $(ffmpeg -version | head -1 | cut -c1-40)" \
  || say "  ✗ 未安装，流式落盘节点无法工作"

say ""
say "完成。下一步："
say "  1) bash models/download_flashvsr.sh $COMFY"
say "  2) bash nodes/install_nodes.sh $COMFY"
say "  3) python patches/patch_lq_bf16.py $COMFY/custom_nodes/ComfyUI_FlashVSR"
