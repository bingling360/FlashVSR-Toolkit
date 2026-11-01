#!/bin/bash
# ============================================================
# FlashVSR 完整方案 - 模型下载
#
# 下载 FlashVSR v1.0 与 FlashVSR+ 全部权重到 ComfyUI 对应目录。
# 每个文件都会校验精确字节数，失败自动重试。
#
# 用法：
#   bash download_flashvsr.sh [ComfyUI根目录]
#   默认 /mnt/workspace/ComfyUI
# ============================================================
set -u

COMFY="${1:-/mnt/workspace/ComfyUI}"
FD="$COMFY/models/FlashVSR"
VD="$COMFY/models/vae"
MS_BASE='https://www.modelscope.cn/models/AI-ModelScope/FlashVSR/resolve/master'
HF_BASE='https://hf-mirror.com/ruchangyao/FlashVSR_plus/resolve/main'
GH_PROXY='https://gh-proxy.com/https://raw.githubusercontent.com'

mkdir -p "$FD" "$VD"

say(){ echo "[$(date +%H:%M:%S)] $*"; }

# dl <url> <目标路径> <期望字节数>
dl(){
  local url="$1" dest="$2" exp="$3" i sz
  for i in 1 2 3 4 5 6 7 8; do
    sz=$(stat -c %s "$dest" 2>/dev/null || echo 0)
    [ "$sz" = "$exp" ] && { say "  ✓ $(basename "$dest")"; return 0; }
    say "  第 $i 轮 $(basename "$dest")  ($sz / $exp)"
    curl -sL -C - -m 1800 -o "$dest" "$url" 2>/dev/null
  done
  sz=$(stat -c %s "$dest" 2>/dev/null || echo 0)
  say "  ✗ $(basename "$dest") 最终 $sz，期望 $exp"
  return 1
}

say "=== FlashVSR v1.0（基础版，6.95 GB）==="
dl "$MS_BASE/LQ_proj_in.ckpt" "$FD/LQ_proj_in.ckpt" 575695268
dl "$MS_BASE/diffusion_pytorch_model_streaming_dmd.safetensors" \
   "$FD/diffusion_pytorch_model_streaming_dmd.safetensors" 5676070392
dl "$MS_BASE/Wan2.1_VAE.pth" "$VD/Wan2.1_VAE.pth" 507609880
dl "$MS_BASE/TCDecoder.ckpt" "$VD/TCDecoder.ckpt" 189018333

say "=== FlashVSR+ turbo（6.83 GB）==="
dl "$HF_BASE/turbo/dit.safetensors" \
   "$FD/turbo_dit_plus.safetensors" 5676070392
dl "$HF_BASE/turbo/lq_proj_in.safetensors" \
   "$FD/lq_proj_in_plus_turbo.safetensors" 1151384272

say "=== 固定条件嵌入（GitHub）==="
dl "$GH_PROXY/OpenImagingLab/FlashVSR/main/examples/WanVSR/prompt_tensor/posi_prompt.pth" \
   "$FD/posi_prompt.pth" 4195504
dl "$GH_PROXY/OpenImagingLab/FlashVSR_Plus/main/prompt_tensor/posi_prompt.pth" \
   "$FD/posi_prompt_plus.pth" 4195504

say "=== TCD 解码器（plus 用，复制一份改名）==="
cp -n "$VD/TCDecoder.ckpt" "$VD/TCDecoder_plus.ckpt" 2>/dev/null || true

say ""
say "=== 结果 ==="
ls -lh "$FD" "$VD" 2>/dev/null | grep -E 'ckpt|safetensors|pth' || true
say "完成。若有个别文件失败，重复执行本脚本会自动续传。"
