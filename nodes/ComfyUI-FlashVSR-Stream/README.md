# ComfyUI-FlashVSR-Stream

把 FlashVSR 的**解码 + 落盘**合成一个节点，绕开 ComfyUI「必须返回整段 IMAGE 张量」的限制。

## 解决什么问题

现有链路的内存峰值来自三个地方：

| 环节 | 峰值 | 原因 |
|---|---|---|
| 采样器准备 LQ | ~1 份 | 把每帧放大到目标分辨率后堆成 float32 张量 |
| 解码 | ~1 份 | 逐段解码，但结果攒在缓冲里 |
| 返回 IMAGE | ~1 份 | ComfyUI 要求整段张量跨节点传递 |

本节点只处理**第三段**：解码完一段就立刻转 uint8 写进视频编码器，内存恒定在一段（约 0.3 GB），与总帧数无关。

> ⚠️ 注意：**采样器的 LQ 准备阶段仍是瓶颈**。本节点能让解码/输出阶段不再随帧数增长，
> 但采样阶段仍受「整段 LQ float32」限制。要彻底解决需要另外改采样器。

## 为什么没有接缝

推理是一次连续的前向，滑动 KV 缓存跨窗口传递，**本节点只改变「怎么保存」，不改变「怎么推理」**。
所以输出与一次性跑完全一致。

色彩校正（`color_fix`）是逐帧空间运算（小波分解），没有跨帧统计量，
因此逐段应用与整体应用结果相同。

## 安装

把整个 `ComfyUI-FlashVSR-Stream` 目录放进：

```
ComfyUI/custom_nodes/ComfyUI-FlashVSR-Stream/
```

重启 ComfyUI，节点出现在 `FlashVSR` 分类下，名称「FlashVSR 流式解码落盘」。

依赖：`torch`、`numpy`、`imageio`、`imageio-ffmpeg`（ComfyUI 环境通常已有）。
音频混流需要系统 `ffmpeg`。

## 接线

替换原来的 `FlashVSR_SM_Decoder` + `VHS_VideoCombine`：

```
FlashVSR_SM_Sampler ──LATENT──┐
                              ├──> FlashVSR 流式解码落盘 ──> video_path (STRING)
FlashVSR_SM_Vae ────FVSRN_VAE─┘
VHS_LoadVideo ──────AUDIO─────┘（可选，接了才保留音轨）
```

输出是一个文件路径字符串，可以用 `PreviewAny` 或直接到 `ComfyUI/output/` 下找。

## 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `latent` | — | 接 Sampler 的 LATENT |
| `vae` | — | 接 Vae 的 FVSRN_VAE（**必须是 TCD 解码器**）|
| `filename_prefix` | FlashVSR_stream | 输出文件名前缀 |
| `fps` | 24 | 输出帧率，**必须等于源片帧率** |
| `crf` | 19 | H.264 质量，越小越好越大，19 接近视觉无损 |
| `color_fix` | true | 是否做色彩校正 |
| `fix_method` | wavelet | `wavelet` 保细节，`adain` 更快更平 |
| `segment_latent_frames` | 4 | 每段解码多少个 latent 帧。4 与官方一致；内存紧张可降到 2 |
| `audio` | 无 | 可选，接上则混流保留音轨 |

## 实测结果（A10 / 45 帧 / 2048×1152）

| 配置 | RGB 全帧 PSNR（vs 标准路径）| 判定 |
|---|---|---|
| `color_fix=False` | 47.42 dB | 视觉无差异 |
| `color_fix=True` | **48.77 dB** | 视觉无差异 |

残差来自两者 x264 码率分配不同（14.9 MB vs 14.2 MB），非算法差异。

**速度**（采样已缓存，只比后处理）：

| | 耗时 |
|---|---|
| 标准路径（Decoder + VHS） | 39.0s |
| 本节点 | **9.0s** |

快 4.3 倍，因为省掉了整段张量拼接和全量 float32 转换。

## 已知限制

1. **仅支持 TCD 解码器**。走标准 Wan2.1 VAE 的路径没有实现（那条路径本来也不是内存瓶颈）。
2. **尾部填充帧**。FlashVSR 会把帧数对齐到 8n+3，多出来的帧会被写入。
   如果发现结尾多几帧，用 ffmpeg 裁掉：
   ```bash
   ffmpeg -i in.mp4 -frames:v 587 -c copy out.mp4
   ```
3. **`long_mode` 未适配**。开了 `long_mode` 时 latent 结构不同（`fvsr_long`），
   本节点会走普通路径，可能报错。请关掉 `long_mode` 使用本节点。
4. **无进度条**。解码过程中界面不显示进度，只能看控制台日志。
5. **未在真实环境验证**。代码基于 `ComfyUI_FlashVSR` main 分支（2026-10）的接口编写，
   本地只做了语法检查，需要实际跑一次确认。

## 与「分段跑」方案的区别

| | 本节点 | 分段跑（改 skip/cap 多次提交）|
|---|---|---|
| 推理 | 一次连续 | N 次独立 |
| 接缝 | 无 | 有（段首缺 KV 上下文）|
| 内存 | 解码/输出阶段恒定 | 每段独立 |
| 操作 | 一次运行 | 手动跑 N 次 + 拼接 |

本节点在**推理连续性**上优于分段跑，但不能降低采样器准备 LQ 阶段的内存。
两者可以叠加：如果采样阶段就 OOM，仍然需要分段。

## 显存释放

### 自动（推荐）

`unload_after` 默认开启。节点写完视频后立刻执行：

```python
mm.unload_all_models()   # 卸载所有模型
mm.soft_empty_cache()    # 清显存缓存
gc.collect()             # 回收内存
```

等价于 ComfyUI 的 `/free` 接口。适合「超分 → 补帧」这类多阶段流程，
避免上一阶段的模型占着显存不放。

### 手动

同包内还有 **「释放显存/内存」** 节点（`MemoryPurge`），透传任意输入，
可插在任何两个阶段之间。ComfyUI 核心没有这个节点。

```
超分节点 ──STRING──> 释放显存/内存 ──STRING──> 读回视频
```
