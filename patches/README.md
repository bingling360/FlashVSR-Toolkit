# 补丁说明

## patch_lq_bf16.py

把 FlashVSR 采样器的 LQ 准备阶段从「攒列表再统一转换」改成「预分配 bf16 缓冲、逐帧写入」。

### 影响范围

| 文件 | 函数 |
|---|---|
| `flashvsr_native/plus_sampler.py` | `prepare_lq_frames` |
| `flashvsr_native/stream_sampler.py` | `prepare_lq_frames` |

### 效果

| 指标 | 补丁前 | 补丁后 |
|---|---|---|
| 峰值内存（552 帧 @2048×1152）| 约 27 GB | **7.8 GB** |
| 画质 | — | **PSNR = inf（逐像素一致）** |
| 速度 | 78.0s | 72.0s |

### 用法

```bash
python patch_lq_bf16.py <path/to/custom_nodes/ComfyUI_FlashVSR>
```

脚本会自动：
1. 备份原文件为 `*.bak_lq_bf16`
2. 精确匹配替换（找不到锚点会报错，不会静默失败）
3. 替换后执行 `py_compile` 语法检查

### 回滚

```bash
cd <ComfyUI_FlashVSR>/flashvsr_native
mv plus_sampler.py.bak_lq_bf16 plus_sampler.py
mv stream_sampler.py.bak_lq_bf16 stream_sampler.py
```

### 注意

节点升级（ComfyUI-Manager 更新）会覆盖补丁，需重打。

---

## 关于解码阶段的补丁

早期还改过 `nodes_native.py::_decode_tcd`（保持 CPU、原地运算、预分配缓冲），
用于缓解标准解码路径的内存峰值。

**这些补丁已被流式落盘节点取代**，如果使用 `ComfyUI-FlashVSR-Stream` 节点，
不需要再改 `nodes_native.py`。

如果坚持用标准 `Decoder` 节点，可以参考这些改法（但不推荐，因为流式节点方案更彻底）：

| 改动 | 作用 |
|---|---|
| `torch.cat(...).to(device)` → 去掉 `.to(device)` | 避免整段视频搬回显存 |
| `del outputs` 紧跟 cat | 及时释放列表 |
| `((frames[0].float() + 1) * 0.5).clamp(0,1)` → `.add_().mul_().clamp_()` | 避免连开 3 份拷贝 |

这些改动会让标准路径的峰值从约 4 份降到约 2 份，但**不能根治**（仍需整段张量）。
