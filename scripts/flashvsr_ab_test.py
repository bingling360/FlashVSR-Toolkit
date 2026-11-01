# -*- coding: utf-8 -*-
"""
FlashVSR 补丁部署 + 质量 A/B 对比

流程：
  1. 用当前（未打 lq 补丁）的代码跑一次基准片  -> baseline.mp4
  2. 应用 patch_lq_bf16.py，重启 ComfyUI
  3. 同样参数再跑一次                        -> patched.mp4
  4. 用 ffmpeg 比较两段的 PSNR / 最大差异

如果补丁在数值上等价，PSNR 应为 inf（逐像素完全一致）。

用法（在实例上执行）：
  python flashvsr_ab_test.py <video_name> [frame_cap]
"""

import json
import os
import subprocess
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8188"
NODE_DIR = "/mnt/workspace/ComfyUI/custom_nodes/ComfyUI_FlashVSR"
OUT_DIR = "/mnt/workspace/ComfyUI/output"


def sh(cmd, timeout=600):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)


def post_prompt(prompt):
    body = json.dumps({"prompt": prompt}).encode()
    req = urllib.request.Request(BASE + "/prompt", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)["prompt_id"]


def wait(pid, limit=1800):
    t0 = time.time()
    while time.time() - t0 < limit:
        try:
            with urllib.request.urlopen(BASE + "/history/" + pid, timeout=15) as r:
                d = json.load(r)
            if d:
                st = d[pid]["status"]
                msg = st["status_str"]
                if msg != "success":
                    for m in st.get("messages", []):
                        if m[0] == "execution_error":
                            msg += " :: " + str(m[1].get("exception_message", ""))[:200]
                return msg, time.time() - t0
        except Exception:
            pass
        time.sleep(3)
    return "timeout", time.time() - t0


def build(video, cap, prefix):
    return {
        "1": {"class_type": "VHS_LoadVideo", "inputs": {
            "video": video, "force_rate": 0, "custom_width": 0, "custom_height": 0,
            "frame_load_cap": cap, "skip_first_frames": 0, "select_every_nth": 1,
            "format": "AnimateDiff"}},
        "2": {"class_type": "FlashVSR_SM_Model", "inputs": {
            "dit": "turbo_dit_plus.safetensors",
            "lq_proj": "lq_proj_in_plus_turbo.safetensors",
            "prompt_embed": "posi_prompt_plus.pth",
            "version": "plus_turbo"}},
        "3": {"class_type": "FlashVSR_SM_Vae", "inputs": {
            "vae": "none", "tcd_encoder": "TCDecoder_plus.ckpt"}},
        "4": {"class_type": "FlashVSR_SM_Sampler", "inputs": {
            "model": ["2", 0], "image": ["1", 0],
            "width": 512, "height": 288, "seed": 0, "scale": 4,
            "kv_ratio": 3.5, "local_range": 11, "sparse_ratio": 2.0,
            "offload": True, "chunk_q": 0, "long_mode": False}},
        "5": {"class_type": "FlashVSR_SM_Decoder", "inputs": {
            "latent": ["4", 0], "vae": ["3", 0],
            "tiled": True, "tile_size": 64, "overlap": 16,
            "temporal_size": 64, "temporal_overlap": 8,
            "color_fix": True, "fix_method": "wavelet"}},
        "6": {"class_type": "VHS_VideoCombine", "inputs": {
            "images": ["5", 0], "audio": ["1", 2], "frame_rate": 24, "loop_count": 0,
            "filename_prefix": prefix, "format": "video/h264-mp4",
            "pix_fmt": "yuv420p", "crf": 0, "save_metadata": False,
            "trim_to_audio": False, "pingpong": False, "save_output": True}},
    }


def newest(prefix):
    cands = [os.path.join(OUT_DIR, f) for f in os.listdir(OUT_DIR)
             if f.startswith(prefix) and f.endswith(".mp4")]
    if not cands:
        return None
    return max(cands, key=os.path.getmtime)


def compare(a, b):
    """用 ffmpeg 的 psnr 滤镜比较两段视频。"""
    cmd = (f'ffmpeg -hide_banner -i "{a}" -i "{b}" '
           f'-lavfi "[0:v][1:v]psnr=stats_file=-" -f null - 2>&1')
    out = sh(cmd).stdout
    for line in out.splitlines():
        if "PSNR" in line and "average" in line:
            return line.strip()
    return out.strip().splitlines()[-1] if out.strip() else "比较失败"


def main():
    video = sys.argv[1] if len(sys.argv) > 1 else "t960.mp4"
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 48

    print("=" * 60)
    print(f"测试片：{video}  帧数上限：{cap}")
    print("=" * 60)

    # ---- 1. 基准 ----
    print("\n[1/4] 跑基准（补丁前）…")
    pid = post_prompt(build(video, cap, "ab_before"))
    msg, dur = wait(pid)
    print(f"      结果：{msg}  {dur:.1f}s")
    before = newest("ab_before")
    if msg != "success" or not before:
        print("基准失败，中止")
        return
    print(f"      输出：{before}")

    # ---- 2. 打补丁 ----
    print("\n[2/4] 应用 lq_bf16 补丁…")
    r = sh(f"/mnt/workspace/comfy_env_cuda/bin/python /mnt/workspace/patch_lq_bf16.py {NODE_DIR}")
    print("      " + (r.stdout or r.stderr).strip().replace("\n", "\n      "))

    print("\n[3/4] 重启 ComfyUI…")
    sh("bash /mnt/workspace/start_gpu.sh", timeout=300)
    for _ in range(40):
        time.sleep(3)
        try:
            urllib.request.urlopen(BASE + "/", timeout=5)
            break
        except Exception:
            pass
    print("      已就绪")

    # ---- 3. 补丁后 ----
    print("\n[4/4] 跑补丁后…")
    pid = post_prompt(build(video, cap, "ab_after"))
    msg, dur = wait(pid)
    print(f"      结果：{msg}  {dur:.1f}s")
    after = newest("ab_after")
    if msg != "success" or not after:
        print("补丁后失败，中止")
        return
    print(f"      输出：{after}")

    # ---- 比较 ----
    print("\n" + "=" * 60)
    print("质量对比（crf=0 无损，排除编码差异）")
    print(compare(before, after))
    print("=" * 60)
    print("PSNR = inf 表示逐像素完全一致；>50dB 视为无可见差异")


if __name__ == "__main__":
    main()
