# -*- coding: utf-8 -*-
"""
FlashVSR Stream Save
====================

把 FlashVSR 采样输出的 LATENT 直接「逐段解码 → 逐段落盘」，不构建整段 IMAGE 张量。

设计要点
--------
* 解码循环本身是逐段进行的（每 4 个 latent 帧一段），本节点复用这个结构，
  每解完一段立刻转 uint8 写入视频编码器，**内存只占一段**。
* 色彩校正是逐帧空间运算（小波分解），逐段做与整体做结果完全一致。
* 因为推理是连续的（KV 缓存跨窗口传递），所以**没有接缝**。
* 输出是视频文件路径，不返回 IMAGE 张量 —— 这是避开 ComfyUI 张量物化的关键。

作者备注
--------
* 依赖：torch, numpy, imageio, imageio-ffmpeg（ComfyUI 环境通常已具备）
* 音频：本节点输出无声视频。如需保留音轨，把 AUDIO 接到 `audio` 输入，
  节点会在写完视频后用 ffmpeg 混流；若环境没有 ffmpeg 会跳过并打印警告。
"""

import os
import gc
import shutil
import subprocess
import tempfile
import time

import numpy as np
import torch


def purge_memory():
    """卸载所有模型 + 清空显存缓存 + 回收内存。

    等价于 ComfyUI 的 /free 接口做的事：
      unload_all_models() -> soft_empty_cache() -> gc.collect()
    """
    freed = []
    try:
        import comfy.model_management as mm
        before = torch.cuda.memory_reserved() / 1073741824 if torch.cuda.is_available() else 0
        mm.unload_all_models()
        mm.soft_empty_cache()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            after = torch.cuda.memory_reserved() / 1073741824
            freed.append(f"显存 {before:.2f}→{after:.2f} GB")
    except Exception as e:
        print(f"[MemoryPurge] 失败：{type(e).__name__}: {e}")
    print(f"[MemoryPurge] 已释放 {' / '.join(freed) if freed else '（无 CUDA）'}")

try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover
    imageio = None


# --------------------------------------------------------------------------
# 色彩校正（逐帧空间运算，从 ComfyUI_FlashVSR/flashvsr_native/color_fix.py 移植）
# --------------------------------------------------------------------------
def _calc_mean_std(feat, eps=1e-5):
    N, C = feat.shape[:2]
    var = feat.view(N, C, -1).var(dim=2, unbiased=False) + eps
    std = var.sqrt().view(N, C, 1, 1)
    mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return mean, std


def _adain(content_feat, style_feat):
    size = content_feat.size()
    style_mean, style_std = _calc_mean_std(style_feat)
    content_mean, content_std = _calc_mean_std(content_feat)
    normalized = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized * style_std.expand(size) + style_mean.expand(size)


def _make_gaussian3x3_kernel(dtype, device):
    vals = [
        [0.0625, 0.125, 0.0625],
        [0.125, 0.25, 0.125],
        [0.0625, 0.125, 0.0625],
    ]
    return torch.tensor(vals, dtype=dtype, device=device)


def _wavelet_blur(x, radius):
    N, C, H, W = x.shape
    base = _make_gaussian3x3_kernel(x.dtype, x.device)
    weight = base.view(1, 1, 3, 3).repeat(C, 1, 1, 1)
    pad = radius
    x_pad = torch.nn.functional.pad(x, (pad, pad, pad, pad), mode="replicate")
    return torch.nn.functional.conv2d(
        x_pad, weight, bias=None, stride=1, padding=0, dilation=radius, groups=C
    )


def _wavelet_decompose(x, levels=5):
    high = torch.zeros_like(x)
    low = x
    for i in range(levels):
        blurred = _wavelet_blur(low, 2 ** i)
        high = high + (low - blurred)
        low = blurred
    return high, low


def _wavelet_reconstruct(content, style, levels=5):
    c_high, _ = _wavelet_decompose(content, levels=levels)
    _, s_low = _wavelet_decompose(style, levels=levels)
    return c_high + s_low


def apply_color_fix(hq_frames, lq_frames, method="wavelet", levels=5, chunk=16):
    """hq/lq: [f,3,H,W] in [0,1]。返回同样形状。逐帧独立，可分多次调用。"""
    f = hq_frames.shape[0]
    outs = []
    for s in range(0, f, chunk):
        e = min(s + chunk, f)
        c = hq_frames[s:e]
        st = lq_frames[s:e]
        if method == "wavelet":
            o = _wavelet_reconstruct(c, st, levels=levels)
        elif method == "adain":
            o = _adain(c, st)
        else:
            raise ValueError("fix_method must be wavelet or adain")
        outs.append(o.clamp_(0, 1))
    return torch.cat(outs, dim=0)


# --------------------------------------------------------------------------
# 视频写出
# --------------------------------------------------------------------------
class _VideoWriter:
    """把 [h,w,3] uint8 帧逐帧通过管道喂给 ffmpeg，参数完全可控。

    不用 imageio：它的 quality/ffmpeg_params 组合会忽略 crf，
    导致编码质量远低于预期（实测 26dB vs 38dB）。
    """

    def __init__(self, path, width, height, fps=24, crf=19, pix_fmt="yuv420p"):
        self.ff = shutil.which("ffmpeg")
        if self.ff is None:
            raise RuntimeError("需要系统 ffmpeg")
        self.path = path
        cmd = [
            self.ff, "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{int(width)}x{int(height)}",
            "-r", str(float(fps)), "-i", "-",
            "-an", "-c:v", "libx264", "-crf", str(int(crf)),
            "-pix_fmt", pix_fmt,
            # 注意：不要加 -colorspace/-color_range 等标记。
            # 实测加上会让 ffmpeg 多做一次范围转换，
            # 往返 PSNR 从 38.4dB 掉到 26dB（均值漂移 1.6 个色阶）。
            path,
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self.count = 0

    def write(self, frames_u8):
        for fr in frames_u8:
            self.proc.stdin.write(fr.tobytes())
            self.count += 1

    def close(self):
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=120)
        except Exception:
            pass


def _mux_audio(video_path, audio, out_path):
    """把 ComfyUI AUDIO 混进视频。失败则把 video 原样复制到 out_path。"""
    ff = shutil.which("ffmpeg")
    if not audio or ff is None:
        shutil.copyfile(video_path, out_path)
        return False

    tmp_wav = tempfile.mktemp(suffix=".wav")
    try:
        wav = audio.get("waveform")
        sr = int(audio.get("sample_rate", 44100))
        if wav is None:
            raise RuntimeError("no waveform")
        if wav.dim() == 3:
            wav = wav[0]
        wav = wav.detach().cpu().float().numpy()
        wav = np.clip(wav, -1.0, 1.0)
        pcm = (wav.T * 32767.0).astype(np.int16)  # [n, ch]
        import wave as wave_mod

        with wave_mod.open(tmp_wav, "wb") as wf:
            wf.setnchannels(pcm.shape[1])
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm.tobytes())

        cmd = [ff, "-y", "-loglevel", "error", "-i", video_path, "-i", tmp_wav,
               "-c:v", "copy", "-c:a", "aac", "-shortest", out_path]
        subprocess.run(cmd, check=True)
        return True
    except Exception as e:
        print(f"[FlashVSR-Stream] 音频混流失败（{type(e).__name__}: {e}），输出无声视频")
        shutil.copyfile(video_path, out_path)
        return False
    finally:
        if os.path.exists(tmp_wav):
            try:
                os.remove(tmp_wav)
            except Exception:
                pass


# --------------------------------------------------------------------------
# 节点
# --------------------------------------------------------------------------
class FlashVSRStreamSave:
    """逐段解码并直接写盘，避免整段 IMAGE 张量导致的内存/显存峰值。"""

    CATEGORY = "FlashVSR"
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "FlashVSR_SM_Sampler 的 LATENT 输出"}),
                "vae": ("FVSRN_VAE", {"tooltip": "FlashVSR_SM_Vae 的 FVSRN_VAE 输出"}),
                "filename_prefix": ("STRING", {"default": "FlashVSR_stream"}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.01}),
                "crf": ("INT", {"default": 19, "min": 0, "max": 51}),
                "color_fix": ("BOOLEAN", {"default": True}),
                "fix_method": (["wavelet", "adain"], {"default": "wavelet"}),
                "segment_latent_frames": ("INT", {
                    "default": 4, "min": 1, "max": 64,
                    "tooltip": "每段解码多少个 latent 帧；4 与官方一致，越小内存越低",
                }),
            },
            "optional": {
                "audio": ("AUDIO", {"tooltip": "可选：接上则混流保留音轨"}),
                "unload_after": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "写完后卸载模型并清空显存缓存。\n"
                               "接补帧等后续阶段时建议保持开启，避免显存残留。",
                }),
            },
        }

    # ---- 工具 ----
    @staticmethod
    def _out_dir():
        try:
            import folder_paths
            d = folder_paths.get_output_directory()
        except Exception:
            d = os.path.join(os.getcwd(), "output")
        os.makedirs(d, exist_ok=True)
        return d

    def execute(self, latent, vae, filename_prefix, fps, crf,
                color_fix, fix_method, segment_latent_frames,
                audio=None, unload_after=True):
        if not isinstance(vae, dict) or vae.get("kind") != "tcd":
            raise ValueError("本节点仅支持 TCD 解码器（FlashVSR_SM_Vae 选 TCDecoder）")

        module = vae["module"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 解码器必须和输入在同一设备上；ComfyUI 可能把它留在 CPU
        try:
            module = module.to(device)
        except Exception:
            pass

        z = latent["fvsr_z"]                       # [1,16,N,h,w] cpu bf16
        LQ = latent.get("fvsr_LQ", None)           # [1,3,F,tH,tW] [-1,1]
        is_plus = bool(latent.get("fvsr_plus", False))
        pad_frames = int(latent.get("fvsr_pad", 0))

        zz = z.to(device=device, dtype=torch.bfloat16)
        lq = LQ.to(device=device, dtype=torch.bfloat16) if LQ is not None else None

        # 输出路径
        out_dir = self._out_dir()
        stamp = time.strftime("%Y%m%d-%H%M%S")
        base = f"{filename_prefix}_{stamp}"
        tmp_path = os.path.join(out_dir, base + "_video.mp4")
        final_path = os.path.join(out_dir, base + "-audio.mp4")
        if not audio:
            final_path = os.path.join(out_dir, base + ".mp4")

        writer = None
        written = 0
        lq_cursor = 0

        try:
            with torch.inference_mode():
                n = zz.shape[2]
                step = max(1, int(segment_latent_frames))

                if is_plus:
                    start = 0
                    while start < n:
                        end = min(start + step, n)
                        first = start == 0
                        frame_start = 0 if first else 4 * start - 3
                        frame_count = 4 * (end - start) - (3 if first else 0)
                        cur = zz[:, :, start:end]
                        cur_lq = lq[:, :, frame_start:frame_start + frame_count] if lq is not None else None
                        out = module.decode_video(cur.transpose(1, 2), parallel=False,
                                                  show_progress_bar=False, cond=cur_lq)
                        frames = self._post(out, cur_lq, color_fix, fix_method)
                        if writer is None:
                            writer = _VideoWriter(tmp_path, frames.shape[2], frames.shape[1],
                                                  fps=fps, crf=crf)
                        writer.write(frames)
                        written += frames.shape[0]
                        del cur, cur_lq, out, frames
                        start = end
                else:
                    total_frames = 4 * n - 3
                    cond = lq[:, :, :total_frames] if lq is not None else None
                    out = module.decode_video(zz.transpose(1, 2), parallel=False,
                                              show_progress_bar=False, cond=cond)
                    frames = self._post(out, cond, color_fix, fix_method)
                    if writer is None:
                        writer = _VideoWriter(tmp_path, frames.shape[2], frames.shape[1],
                                              fps=fps, crf=crf)
                    writer.write(frames)
                    written += frames.shape[0]
                    del out, frames, cond
        finally:
            if writer is not None:
                writer.close()
            torch.cuda.empty_cache()

        # 尾部补帧裁剪：写入时可能多写了 pad_frames 帧，这里不重写文件，
        # 交由调用方用 ffmpeg 截断（正常情况下 pad_frames 为 0）。
        if pad_frames > 0:
            print(f"[FlashVSR-Stream] 注意：有 {pad_frames} 帧尾部填充，如需精确时长请用 ffmpeg 裁剪")

        if audio:
            _mux_audio(tmp_path, audio, final_path)
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        else:
            os.replace(tmp_path, final_path)

        print(f"[FlashVSR-Stream] 完成：{written} 帧 -> {final_path}")

        # 卸载模型 + 清显存，给后续阶段（如补帧）腾地方
        if unload_after:
            purge_memory()

        # 给界面可见的输出：文本显示路径 + 视频预览
        rel = os.path.basename(final_path)
        return {
            "ui": {
                "text": [f"已保存 {written} 帧\n{final_path}"],
                "gifs": [{
                    "filename": rel,
                    "subfolder": "",
                    "type": "output",
                    "format": "video/h264-mp4",
                }],
            },
            "result": (final_path,),
        }

    # ---- 单段后处理：[-1,1] → color fix → uint8 ----
    @staticmethod
    def _post(out, cond_lq, color_fix, fix_method):
        o = out.transpose(1, 2)[0]              # [3,f,h,w] in [0,1]
        o = o.float()
        if color_fix and cond_lq is not None:
            hq = o.permute(1, 0, 2, 3).contiguous()          # [f,3,h,w]
            st = ((cond_lq[0].float() + 1.0) * 0.5).clamp(0, 1)  # [3,f,h,w]
            st = st.permute(1, 0, 2, 3).contiguous()          # [f,3,h,w]
            if st.shape[0] < hq.shape[0]:
                pad = hq.shape[0] - st.shape[0]
                st = torch.cat([st, st[-1:].repeat(pad, 1, 1, 1)], dim=0)
            st = st[:hq.shape[0]]
            if st.shape[-2:] != hq.shape[-2:]:
                st = torch.nn.functional.interpolate(
                    st.permute(0, 3, 1, 2), size=hq.shape[-2:],
                    mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
            hq = apply_color_fix(hq, st, method=fix_method)
            o = hq.permute(1, 0, 2, 3)
        # DEBUG: 转储首帧，用于和标准路径对比
        try:
            if os.environ.get("FVSR_DEBUG_DUMP"):
                _f0 = o[:, 0].detach().float().cpu()
                if not os.path.exists("/tmp/dbg_node_f0.pt"):
                    torch.save(_f0, "/tmp/dbg_node_f0.pt")
                    print(f"[DBG] node f0 shape={tuple(_f0.shape)} "
                          f"min={float(_f0.min()):.4f} max={float(_f0.max()):.4f}")
        except Exception as _e:
            print("[DBG] node dump failed", _e)
        u8 = (o.clamp(0, 1) * 255.0).round().to(torch.uint8)
        return u8.permute(1, 2, 3, 0).contiguous().cpu().numpy()  # [f,h,w,3]


class MemoryPurge:
    """独立的内存/显存清理节点，透传任意输入。

    放在两个重负载阶段之间（例如超分 → 补帧），确保前一个阶段的模型和缓存
    完全释放后再加载下一个。ComfyUI 核心没有这个节点，这里补上。
    """

    CATEGORY = "FlashVSR"
    FUNCTION = "run"
    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("output",)
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("*", {"tooltip": "任意类型的输入，原样传出"}),
            },
        }

    def run(self, value):
        purge_memory()
        return (value,)


NODE_CLASS_MAPPINGS = {
    "FlashVSRStreamSave": FlashVSRStreamSave,
    "MemoryPurge": MemoryPurge,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FlashVSRStreamSave": "FlashVSR 流式解码落盘",
    "MemoryPurge": "释放显存/内存",
}
