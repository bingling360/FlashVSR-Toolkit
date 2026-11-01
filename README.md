# FlashVSR 优化包

针对 **ComfyUI + FlashVSR / FlashVSR+** 在 Linux（NVIDIA A10，24GB 显存 / 28GB 内存）上的
部署与内存优化。解决的核心问题是：**长视频超分时显存和内存必然爆掉**。

---

## 这个包解决什么问题

FlashVSR 是流式视频超分模型，但 ComfyUI 的节点接口要求「一个节点输出一个完整张量」，
导致流式结果被强行物化成整段 `[F,H,W,C]` 张量。帧数一多就爆：

| 阶段 | 原始峰值（552 帧 @2048×1152）| 优化后 |
|---|---|---|
| 采样器准备 LQ | 约 27 GB | **7.8 GB** |
| 解码 + 输出 | 约 15.6 GB | **0.3 GB**（恒定）|
| **合计** | **约 42 GB（必爆）** | **约 9 GB** |

---

## 三个组成部分

### 1. 流式解码落盘节点（`nodes/`）

替换原来的 `Decoder` + `VHS_VideoCombine`。解码完一段立刻写进视频文件，不构建整段张量。

- 画质：与标准路径 **PSNR 48.77 dB**（视觉无差异）
- 速度：后处理 **9.0s** vs 标准路径 39.0s（快 4.3 倍）
- 内存：恒定约 0.3 GB，与帧数无关

### 2. LQ 准备阶段补丁（`patches/patch_lq_bf16.py`）

把采样器里「先攒列表再统一转换」改成「预分配 bf16 缓冲、逐帧写入」。

- 画质：**PSNR = inf**（逐像素完全一致，已验证）
- 内存：27 GB → 7.8 GB

### 3. Block-Sparse-Attention 编译方案（`scripts/build_bsa.sh` + `docs/BUILD-BSA.md`）

FlashVSR 必须依赖这个 CUDA 扩展，否则会退化成稠密 mask 直接爆显存
（实测请求 94.92 GiB 的单个张量）。Linux 没有现成 wheel，必须自己编译。

> 编译产物 `block_sparse_attn_cuda.cpython-312-x86_64-linux-gnu.so` 有 **508 MB**，
> 不适合放进 git。请用脚本自行编译，坑位记录见 `docs/BUILD-BSA.md`。

---

## 目录结构

```
FlashVSR-优化包/
├── nodes/
│   └── ComfyUI-FlashVSR-Stream/     流式解码落盘节点
│       ├── __init__.py
│       └── README.md
├── patches/
│   └── patch_lq_bf16.py             LQ 阶段 bf16 预分配补丁
├── scripts/
│   ├── build_bsa.sh                 Block-Sparse-Attention 编译脚本
│   ├── flashvsr_ab_test.py          补丁前后画质 A/B 测试
│   └── clean_check.py               节点 vs 标准路径的干净对比测试
├── workflows/
│   └── 5-超分-FlashVSR-流式.json    完整工作流
└── docs/
    ├── BUILD-BSA.md                 编译踩坑记录
    └── FINDINGS.md                  原理分析与实测数据
```

---

## 快速开始

### 前提

- ComfyUI（0.42+）
- 已安装 [smthemex/ComfyUI_FlashVSR](https://github.com/smthemex/ComfyUI_FlashVSR)
- 已下载 FlashVSR+ 权重（`turbo_dit_plus` / `lq_proj_in_plus_turbo` / `posi_prompt_plus` / `TCDecoder_plus`）
- 已编译 `block_sparse_attn`（见 `docs/BUILD-BSA.md`）

### 安装节点

```bash
cp -r nodes/ComfyUI-FlashVSR-Stream ComfyUI/custom_nodes/
```

重启 ComfyUI，节点出现在 `FlashVSR` 分类下。

### 打补丁

```bash
python patches/patch_lq_bf16.py ComfyUI/custom_nodes/ComfyUI_FlashVSR
```

会同时修改 `plus_sampler.py` 和 `stream_sampler.py`，并自动备份为 `.bak_lq_bf16`。

### 使用工作流

把 `workflows/5-超分-FlashVSR-流式.json` 拖进 ComfyUI，或放到
`ComfyUI/user/default/workflows/`。

---

## 权重与下载来源

| 文件 | 来源 | 大小 |
|---|---|---|
| v1.0 全套 | ModelScope `AI-ModelScope/FlashVSR` | 6.95 GB |
| plus turbo 套件 | hf-mirror `ruchangyao/FlashVSR_plus` | 6.83 GB |
| `posi_prompt_plus.pth` | GitHub `OpenImagingLab/FlashVSR_Plus` | 4.2 MB |
| `TCDecoder.ckpt` | 同 plus 仓库，复制为 `TCDecoder_plus.ckpt` 放 `models/vae` | 189 MB |

ModelScope 上没有 FlashVSR+ 仓库，只有基础版。

---

## 实测数据汇总

环境：NVIDIA A10（24GB）、28GB 内存、CUDA 13.0、PyTorch 2.13、Python 3.12

| 场景 | 分辨率 | 帧数 | 耗时 | 备注 |
|---|---|---|---|---|
| 短片段 | 2048×1152 | 48 | 54.1s | 含模型加载 |
| 全流程 | 2048×1152 | 48 | 45.0s | 模型已缓存 |
| 后处理对比 | 2048×1152 | 45 | 9.0s vs 39.0s | 流式节点 vs 标准路径 |

约 **1.1 秒/帧**，比 SeedVR2 7B 的 4 秒/帧快约 4 倍。

---

## 已知限制

1. **采样阶段仍有上限**。补丁把 LQ 准备从 27 GB 降到 7.8 GB，但仍随帧数线性增长。
   2048×1152 下单次约能跑 550 帧，更长的片子仍需分段。
2. **仅支持 TCD 解码器**。标准 Wan2.1 VAE 路径未适配。
3. **不支持 `long_mode`**。开了会走错分支。
4. **节点更新会覆盖补丁**。升级 `ComfyUI_FlashVSR` 后需重打。
5. **尺寸必须对齐 128**。输出 = `width×scale` 向下取整到 128 倍数后居中裁剪，
   16:9 推荐 `512×288 + scale 4 → 2048×1152`。

---

## 致谢

- [OpenImagingLab/FlashVSR](https://github.com/OpenImagingLab/FlashVSR) — 模型与论文
- [smthemex/ComfyUI_FlashVSR](https://github.com/smthemex/ComfyUI_FlashVSR) — ComfyUI 封装
- [mit-han-lab/Block-Sparse-Attention](https://github.com/mit-han-lab/Block-Sparse-Attention) — 稀疏注意力后端
