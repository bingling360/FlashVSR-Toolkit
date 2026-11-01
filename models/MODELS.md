# 模型清单与下载地址

所有地址均已实测可用（2026-11，阿里云 DSW + NVIDIA A10 环境）。

---

## 一、FlashVSR（本方案主体）

### 一键下载

```bash
bash models/download_flashvsr.sh /mnt/workspace/ComfyUI
```

脚本会校验每个文件的**精确字节数**，失败自动重试续传。

### 文件明细

#### FlashVSR v1.0 — ModelScope

来源：`https://www.modelscope.cn/models/AI-ModelScope/FlashVSR`

| 文件 | 放置位置 | 大小（字节）|
|---|---|---|
| `LQ_proj_in.ckpt` | `models/FlashVSR/` | 575,695,268 |
| `diffusion_pytorch_model_streaming_dmd.safetensors` | `models/FlashVSR/` | 5,676,070,392 |
| `Wan2.1_VAE.pth` | `models/vae/` | 507,609,880 |
| `TCDecoder.ckpt` | `models/vae/` | 189,018,333 |

#### FlashVSR+ turbo — hf-mirror

来源：`https://hf-mirror.com/ruchangyao/FlashVSR_plus`
（HuggingFace 原仓库 `ruchangyao/FlashVSR_plus`，国内用 hf-mirror 镜像）

| 仓库内路径 | 重命名为 | 大小（字节）|
|---|---|---|
| `turbo/dit.safetensors` | `models/FlashVSR/turbo_dit_plus.safetensors` | 5,676,070,392 |
| `turbo/lq_proj_in.safetensors` | `models/FlashVSR/lq_proj_in_plus_turbo.safetensors` | 1,151,384,272 |

> 还有 `base/` 目录下的 base 版权重（质量优先、速度约慢 2 倍），按需下载。

#### 固定条件嵌入 — GitHub

| 文件 | 来源 | 大小 |
|---|---|---|
| `posi_prompt.pth` | `OpenImagingLab/FlashVSR` → `examples/WanVSR/prompt_tensor/` | 4,195,504 |
| `posi_prompt_plus.pth` | `OpenImagingLab/FlashVSR_Plus` → `prompt_tensor/` | 4,195,504 |

GitHub 直连不通时用 gh-proxy 前缀：`https://gh-proxy.com/https://raw.githubusercontent.com/...`

#### TCD 解码器

plus 模式必须用 `TCDecoder_plus.ckpt`。内容与 `TCDecoder.ckpt` 相同，复制改名即可：

```bash
cp models/vae/TCDecoder.ckpt models/vae/TCDecoder_plus.ckpt
```

### 目录最终形态

```
ComfyUI/models/
├── FlashVSR/
│   ├── LQ_proj_in.ckpt                          575 MB
│   ├── diffusion_pytorch_model_streaming_dmd.safetensors   5.3 GB
│   ├── turbo_dit_plus.safetensors               5.3 GB
│   ├── lq_proj_in_plus_turbo.safetensors        1.1 GB
│   ├── posi_prompt.pth                          4.0 MB
│   ├── posi_prompt_plus.pth                     4.0 MB
│   └── TCDecoder_plus.ckpt                      181 MB
└── vae/
    ├── Wan2.1_VAE.pth                           485 MB
    ├── TCDecoder.ckpt                           181 MB
    └── TCDecoder_plus.ckpt                      181 MB
```

---

## 二、SeedVR2（对比方案）

来源：节点 `numz/ComfyUI-SeedVR2_VideoUpscaler` 首次运行时自动从 ModelScope 下载。

| 文件 | 大小 | 用途 |
|---|---|---|
| `seedvr2_ema_7b_fp8_e4m3fn.safetensors` | 7.7 GB | 7B 主模型，画质最好，约 4 秒/帧 |
| `seedvr2_ema_3b_fp8_e4m3fn.safetensors` | 3.2 GB | 3B 主模型，约 1.5 秒/帧 |
| `ema_vae_fp16.safetensors` | 479 MB | VAE |

放置位置：`models/SEEDVR2/`

---

## 三、GMFSS（补帧）

来源：节点 `ComfyUI-Frame-Interpolation` 首次运行时自动从 HuggingFace 下载。

若自动下载失败，手动下载 `GMFSS_fortuna` 系列权重放到节点目录的 `ckpts/` 下。
具体文件名以节点日志提示为准。

实测速度：1376×768 / 24fps 输入，**0.63 秒/输入帧**（比 rife 慢约 6 倍，质量更好）。

---

## 四、网络环境说明

这台实例的网络实测结果（2026-11-01）：

| 目标 | 状态 | 速度 |
|---|---|---|
| ModelScope | ✅ 通 | 约 15 MB/s |
| hf-mirror.com | ✅ 通 | 约 10 MB/s |
| huggingface.co | ❌ 不通 | — |
| github.com | ⚠️ 极慢 | 约 28 KB/s |
| codeload.github.com | ✅ 通 | 可用（下载 zip） |
| gh-proxy.com | ✅ 通 | 可用（raw 文件） |

**结论**：模型走 ModelScope / hf-mirror，代码走 codeload，单文件走 gh-proxy。
