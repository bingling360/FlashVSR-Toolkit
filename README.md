# FlashVSR Toolkit for ComfyUI

Memory-optimized deployment of [FlashVSR / FlashVSR+](https://github.com/OpenImagingLab/FlashVSR)
video super-resolution in ComfyUI.

FlashVSR is a streaming model, but ComfyUI's node contract forces the entire
`[F,H,W,C]` tensor into memory — so long videos always OOM. This toolkit fixes that.

| | Before | After |
|---|---|---|
| Peak memory, 552 frames @ 2048×1152 | ~42 GB (OOM) | **~9 GB** |
| Post-processing time, 45 frames | 39.0s | **9.0s** |

Output is verified equivalent to the standard path: the LQ patch is **bit-identical
(PSNR = inf)**, and the streaming node reaches **PSNR 48.8 dB** (visually indistinguishable).

> ⚠️ **Not suitable for 2D animation / line art.** FlashVSR is a generative diffusion
> model trained on real-world video. On crisp 2D line art it smooths edges and
> hallucinates texture — measured PSNR 19.2 dB vs. the source, i.e. *worse* than not
> upscaling. Use Real-ESRGAN (anime) or waifu2x for that content.

**Contents**: streaming decode-to-disk node · bf16 LQ preparation patch ·
Block-Sparse-Attention build guide · model download scripts · 5 ready-to-use workflows.

```bash
bash env/setup_env.sh /path/to/ComfyUI          # dependency check + known-issue fixes
bash env/build_bsa.sh                           # required, ~70 min compile
bash models/download_flashvsr.sh /path/to/ComfyUI
bash nodes/install_nodes.sh /path/to/ComfyUI
python patches/patch_lq_bf16.py /path/to/ComfyUI/custom_nodes/ComfyUI_FlashVSR
```

Then drop any JSON from `workflows/` into the ComfyUI canvas.

---

# FlashVSR 完整方案（中文）

在 ComfyUI 上部署 **FlashVSR / FlashVSR+ 视频超分**，并解决长视频的内存爆炸问题。
附带 SeedVR2、GMFSS 补帧的配套工作流。

环境：NVIDIA A10（24GB 显存 / 28GB 内存）、CUDA 13.0、PyTorch 2.13、Python 3.12

---

## ⚠️ 先读这一段：适用内容

**FlashVSR 是生成式扩散模型，训练数据是真实世界视频。**

| 内容类型 | 适用性 |
|---|---|
| 实拍视频（噪点、压缩伪影、轻微模糊）| ✅ 效果好，这是它的强项 |
| 屏幕录制、UI 演示 | ✅ 文字锐利度提升明显 |
| **2D 动画 / 线稿 / 平涂** | ❌ **不要用** |

对 2D 动画实测：1376×768 的清晰线稿，超分后 PSNR 只有 **19.23 dB**，
线条被柔化、色块边缘发糊——**比原片更差**。原因是模型把锐利线条当成噪声去平滑了。

2D 内容请用 Real-ESRGAN anime / waifu2x 这类只放大不重绘的模型。

---

## 这个包解决什么问题

FlashVSR 模型本身是**流式**的（逐窗口推进），但 ComfyUI 的节点接口要求
「一个节点输出一个完整张量」，导致流式结果被强行物化成整段 `[F,H,W,C]` 张量。
帧数一多必爆：

| 阶段 | 原始峰值（552 帧 @2048×1152）| 优化后 |
|---|---|---|
| 采样器准备 LQ | 约 27 GB | **7.8 GB** |
| 解码 + 输出 | 约 15.6 GB | **0.3 GB**（恒定）|
| **合计** | **约 42 GB（必爆）** | **约 9 GB** |

---

## 目录结构

```
FlashVSR-优化包/
├── env/
│   ├── setup_env.sh              依赖检查 + 修复 transformer_engine
│   └── build_bsa.sh              编译 Block-Sparse-Attention（必需，约 70 分钟）
├── models/
│   ├── download_flashvsr.sh      一键下载全部权重（带字节校验）
│   └── MODELS.md                 地址清单、大小、网络环境说明
├── nodes/
│   ├── install_nodes.sh          安装全部自定义节点
│   └── ComfyUI-FlashVSR-Stream/  自制节点（流式落盘 + 显存释放）
├── patches/
│   ├── patch_lq_bf16.py          LQ 阶段 bf16 预分配
│   └── README.md                 影响范围与回滚
├── scripts/
│   ├── flashvsr_ab_test.py       补丁前后画质 A/B 测试
│   └── clean_check.py            节点 vs 标准路径对比测试
├── workflows/
│   ├── 1-GMFSS-VFI.json                纯补帧
│   ├── 2-SeedVR2-SR.json               SeedVR2 超分
│   ├── 4-FlashVSR-Plus.json            FlashVSR+ 超分（标准节点）
│   ├── 5-FlashVSR-Stream.json          FlashVSR+ 超分（流式节点）
│   └── 6-Upscale-Interp-Combo.json     超分 + 补帧一体化
└── docs/
    ├── BUILD-BSA.md              编译踩坑记录（四个关键坑）
    └── FINDINGS.md               原理分析与实测数据
```

---

## 部署顺序

```bash
# 0. 环境检查（会顺便修好 transformer_engine 导致的节点加载失败）
bash env/setup_env.sh /mnt/workspace/ComfyUI

# 1. 编译稀疏注意力后端（必需，约 70 分钟）
bash env/build_bsa.sh

# 2. 下载模型（约 14 GB）
bash models/download_flashvsr.sh /mnt/workspace/ComfyUI

# 3. 安装节点
bash nodes/install_nodes.sh /mnt/workspace/ComfyUI

# 4. 打 LQ 补丁
python patches/patch_lq_bf16.py /mnt/workspace/ComfyUI/custom_nodes/ComfyUI_FlashVSR

# 5. 重启 ComfyUI，把 workflows/ 里的 json 拖进界面
```

---

## 三块核心改动

### 1. 流式解码落盘节点

替换原来的 `Decoder` + `VHS_VideoCombine`。解码完一段立刻写进视频文件，
不构建整段张量。

- 画质：与标准路径 **PSNR 48.77 dB**（视觉无差异）
- 速度：后处理 **9.0s** vs 标准路径 39.0s（快 4.3 倍）
- 内存：恒定约 0.3 GB，与帧数无关
- `unload_after` 默认开启：写完自动卸载模型 + 清显存，给后续阶段腾地方

另附 **「释放显存/内存」节点**（`MemoryPurge`），ComfyUI 核心没有这个，可插在任意两阶段之间。

### 2. LQ 准备阶段补丁

把「先攒列表再统一转换」改成「预分配 bf16 缓冲、逐帧写入」。

- 画质：**PSNR = inf**（逐像素完全一致）
- 内存：27 GB → 7.8 GB

### 3. Block-Sparse-Attention 编译方案

不装会退化成稠密 mask，请求 **94.92 GiB** 的单个张量直接爆显存。
Linux 没有现成 wheel，必须源码编译。四个坑详见 `docs/BUILD-BSA.md`：

| 坑 | 解法 |
|---|---|
| codeload 的 zip 不含 submodule | 单独下 CUTLASS 补进 `csrc/cutlass/` |
| CUTLASS 版本必须精确匹配 | 用锁定的 commit `a75b4ac`，别用 release tag |
| `MAX_JOBS=2` 会被 OOM killer 杀掉 | 必须 `MAX_JOBS=1` |
| 验证时 `libc10.so` 找不到 | 先 `import torch` 再 `import block_sparse_attn` |

> 编译产物 `.so` 有 508 MB，不适合入库，请自行编译。

---

## 参数速查

### 尺寸规则（最容易踩的坑）

`width`/`height` 是**送进模型前的低清尺寸**，不是输出尺寸：

```
输出宽 = (width  × scale) 向下取整到 128 的倍数
输出高 = (height × scale) 向下取整到 128 的倍数
多余部分被居中裁掉
净放大倍数 = 4 × (LQ ÷ 源分辨率)
```

| 源分辨率 | LQ | scale | 输出 | 降采样 | 说明 |
|---|---|---|---|---|---|
| 1376×768 | 512×288 | 4 | 2048×1152 | 2.69× | 丢 86% 像素，画质容易不如原片 |
| 1376×768 | 704×384 | 2 | 2816×1536 | 2× | 折中 |
| 1376×768 | 1408×768 | 1 | 1408×768 | 无 | 纯修复，但超出训练分布 |

**如果成品不如原片，先把 `scale` 降到 1-2、把 `width/height` 提到接近源分辨率。**

### 采样器

| 参数 | 默认 | 作用 | 调法 |
|---|---|---|---|
| `scale` | 4 | 放大倍数（模型按 4× 训练）| 不建议改 |
| `kv_ratio` | 3.5 | 滑动 KV 窗口长度 | 闪烁明显调到 5-7 |
| `local_range` | 11 | 局部注意力范围 | 9 更锐利，11 更稳定 |
| `sparse_ratio` | 2.0 | 稀疏保留比例 | 降到 1.0 可加速 20-30% |
| `offload` | true | 采样后卸 DiT | 长视频建议开 |
| `chunk_q` | 0 | 查询分块（128 倍数）| OOM 时设 128 |

---

## 实测数据

| 场景 | 分辨率 | 帧数 | 耗时 |
|---|---|---|---|
| 短片段（含模型加载）| 2048×1152 | 48 | 54.1s |
| 完整流程 | 2048×1152 | 48 | 45.0s |
| 后处理对比 | 2048×1152 | 45 | 9.0s vs 39.0s |

约 **1.1 秒/帧**，比 SeedVR2 7B 的 4 秒/帧快约 4 倍。

GMFSS 补帧：1376×768 / 24fps 输入，**0.63 秒/输入帧**。

---

## 已知限制

1. **采样阶段仍有上限**。补丁把 LQ 准备从 27 GB 降到 7.8 GB，但仍随帧数线性增长。
   2048×1152 下单次约能跑 550 帧，更长的片子需分段。
2. **仅支持 TCD 解码器**（VAE 节点选 `TCDecoder_plus.ckpt`）。
3. **不支持 `long_mode`**，开了会走错分支。
4. **节点更新会覆盖补丁**。升级 `ComfyUI_FlashVSR` 后需重打。
5. **2D 动画不适用**，见开头。

---

## 致谢

- [OpenImagingLab/FlashVSR](https://github.com/OpenImagingLab/FlashVSR) — 模型与论文
- [smthemex/ComfyUI_FlashVSR](https://github.com/smthemex/ComfyUI_FlashVSR) — ComfyUI 封装
- [mit-han-lab/Block-Sparse-Attention](https://github.com/mit-han-lab/Block-Sparse-Attention) — 稀疏注意力后端
