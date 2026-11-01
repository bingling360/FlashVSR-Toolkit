#!/bin/bash
# ============================================================
# 安装所需的自定义节点
#
# 用法： bash install_nodes.sh [ComfyUI根目录]
# ============================================================
set -u

COMFY="${1:-/mnt/workspace/ComfyUI}"
CN="$COMFY/custom_nodes"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$CN"

say(){ echo "[$(date +%H:%M:%S)] $*"; }

# GitHub 直连很慢时用 codeload 下载 zip
fetch_repo(){
  local repo="$1" dir="$2" branch="${3:-main}"
  if [ -d "$CN/$dir" ]; then
    say "  跳过（已存在）$dir"
    return 0
  fi
  say "  下载 $repo ..."
  local tmp; tmp=$(mktemp -d)
  if curl -sL -m 300 -o "$tmp/r.zip" "https://codeload.github.com/$repo/zip/refs/heads/$branch"; then
    if command -v unzip >/dev/null; then
      unzip -q "$tmp/r.zip" -d "$tmp/x"
    else
      python3 -c "import zipfile,sys;zipfile.ZipFile('$tmp/r.zip').extractall('$tmp/x')"
    fi
    local inner; inner=$(ls -d "$tmp"/x/*/ 2>/dev/null | head -1)
    if [ -n "$inner" ]; then
      mv "$inner" "$CN/$dir"
      say "  ✓ $dir"
    else
      say "  ✗ 解压结构异常"
    fi
  else
    say "  ✗ 下载失败：$repo"
  fi
  rm -rf "$tmp"
}

say "=== 1. FlashVSR 官方封装（必需）==="
fetch_repo "smthemex/ComfyUI_FlashVSR" "ComfyUI_FlashVSR"

say "=== 2. VideoHelperSuite（视频读写，必需）==="
fetch_repo "Kosinkadink/ComfyUI-VideoHelperSuite" "ComfyUI-VideoHelperSuite"

say "=== 3. Frame Interpolation（补帧，可选）==="
fetch_repo "Fannovel16/ComfyUI-Frame-Interpolation" "ComfyUI-Frame-Interpolation"

say "=== 4. SeedVR2（对比方案，可选）==="
fetch_repo "numz/ComfyUI-SeedVR2_VideoUpscaler" "ComfyUI-SeedVR2_VideoUpscaler"

say "=== 5. 本方案的自制节点 ==="
if [ -d "$CN/ComfyUI-FlashVSR-Stream" ]; then
  say "  跳过（已存在）"
else
  cp -r "$HERE/ComfyUI-FlashVSR-Stream" "$CN/"
  say "  ✓ ComfyUI-FlashVSR-Stream"
fi

say ""
say "完成。重启 ComfyUI 后生效。"
say "注意：第三方节点首次运行时会自动下载各自的模型。"
