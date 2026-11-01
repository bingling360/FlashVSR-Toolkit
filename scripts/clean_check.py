import json
import os
import subprocess
import time
import urllib.request

BASE = "http://127.0.0.1:8188"
OUT = "/mnt/workspace/ComfyUI/output"
TAG = "chk"


def base_nodes():
    return {
        "1": {"class_type": "VHS_LoadVideo", "inputs": {
            "video": "t960.mp4", "force_rate": 0, "custom_width": 0, "custom_height": 0,
            "frame_load_cap": 48, "skip_first_frames": 0, "select_every_nth": 1,
            "format": "AnimateDiff"}},
        "2": {"class_type": "FlashVSR_SM_Model", "inputs": {
            "dit": "turbo_dit_plus.safetensors",
            "lq_proj": "lq_proj_in_plus_turbo.safetensors",
            "prompt_embed": "posi_prompt_plus.pth", "version": "plus_turbo"}},
        "3": {"class_type": "FlashVSR_SM_Vae", "inputs": {
            "vae": "none", "tcd_encoder": "TCDecoder_plus.ckpt"}},
        "4": {"class_type": "FlashVSR_SM_Sampler", "inputs": {
            "model": ["2", 0], "image": ["1", 0],
            "width": 512, "height": 288, "seed": 0, "scale": 4,
            "kv_ratio": 3.5, "local_range": 11, "sparse_ratio": 2.0,
            "offload": True, "chunk_q": 0, "long_mode": False}},
    }


def run(prompt, tag):
    body = json.dumps({"prompt": prompt}).encode()
    req = urllib.request.Request(BASE + "/prompt", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        r = json.load(urllib.request.urlopen(req, timeout=30))
    except urllib.error.HTTPError as e:
        print(tag, "SUBMIT_ERROR", e.read().decode()[:400])
        return None
    pid = r["prompt_id"]
    t0 = time.time()
    while time.time() - t0 < 1200:
        try:
            with urllib.request.urlopen(BASE + "/history/" + pid, timeout=15) as rr:
                d = json.load(rr)
            if d:
                st = d[pid]["status"]
                msg = st["status_str"]
                if msg != "success":
                    for m in st.get("messages", []):
                        if m[0] == "execution_error":
                            msg += " :: " + str(m[1].get("exception_message", ""))[:250]
                print(f"  {tag}: {msg} {time.time()-t0:.1f}s")
                break
        except Exception:
            pass
        time.sleep(3)
    cands = [os.path.join(OUT, f) for f in os.listdir(OUT)
             if f.startswith(tag) and f.endswith(".mp4")]
    return max(cands, key=os.path.getmtime) if cands else None


# 清理旧文件
for f in os.listdir(OUT):
    if f.startswith(("iso_", "chk_", "node_test", "ab_")):
        try:
            os.remove(os.path.join(OUT, f))
        except Exception:
            pass
print("已清理旧输出")

# 标准路径
p = base_nodes()
p["5"] = {"class_type": "FlashVSR_SM_Decoder", "inputs": {
    "latent": ["4", 0], "vae": ["3", 0], "tiled": True, "tile_size": 64,
    "overlap": 16, "temporal_size": 64, "temporal_overlap": 8,
    "color_fix": False, "fix_method": "wavelet"}}
p["6"] = {"class_type": "VHS_VideoCombine", "inputs": {
    "images": ["5", 0], "audio": ["1", 2], "frame_rate": 24, "loop_count": 0,
    "filename_prefix": "chk_std", "format": "video/h264-mp4",
    "pix_fmt": "yuv420p", "crf": 0, "save_metadata": False,
    "trim_to_audio": False, "pingpong": False, "save_output": True}}
a = run(p, "chk_std")

# 流式节点
p = base_nodes()
p["5"] = {"class_type": "FlashVSRStreamSave", "inputs": {
    "latent": ["4", 0], "vae": ["3", 0],
    "filename_prefix": "chk_node", "fps": 24.0, "crf": 0,
    "color_fix": False, "fix_method": "wavelet",
    "segment_latent_frames": 4, "audio": ["1", 2]}}
b = run(p, "chk_node")

print()
if not a or not b:
    print("缺少输出:", a, b)
    raise SystemExit

for f in (a, b):
    sz = os.path.getsize(f) / 1048576
    info = subprocess.run(
        f'ffprobe -v error -select_streams v:0 -show_entries stream=width,height,nb_frames '
        f'-of csv=p=0 "{f}"', shell=True, capture_output=True, text=True).stdout.strip()
    print(f"  {os.path.basename(f)}  {sz:.1f}MB  {info}")

cmd = (f'ffmpeg -hide_banner -i "{a}" -i "{b}" '
       f'-lavfi "[0:v]format=rgb24[x];[1:v]format=rgb24[y];[x][y]psnr" -f null - 2>&1')
out = subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout
line = [l for l in out.splitlines() if "PSNR" in l]
print("\nRGB 全帧对比:", line[-1].strip() if line else out[-300:])
