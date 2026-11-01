# 编译 Block-Sparse-Attention（Linux 踩坑记录）

FlashVSR 依赖 [mit-han-lab/Block-Sparse-Attention](https://github.com/mit-han-lab/Block-Sparse-Attention)
提供块级稀疏注意力。**不装会直接爆显存**：

```
Requested : 94.92 GiB      ← 单个张量
Device limit : 22.18 GiB
```

原因是 `flashvsr_native/sparse_attn.py` 里的 `_sdpa_with_token_mask` 回退实现，
在没有 `block_sparse_attn` 时会把稀疏掩码展开成稠密矩阵。

Linux 没有现成 wheel（社区发布的都是 `win_amd64`），必须从源码编译。

---

## 环境

```
Ubuntu / CUDA 13.0 / PyTorch 2.13.0+cu130 / Python 3.12 / sm86 (A10)
编译耗时：约 70 分钟（单进程）
产物大小：508 MB（block_sparse_attn_cuda.cpython-312-x86_64-linux-gnu.so）
```

---

## 四个关键坑

### 坑 1：codeload 下载的 zip 不含 submodule

BSA 依赖 CUTLASS，但它是以 git submodule 的形式引用的。用 `codeload.github.com/.../zip`
下载的包**不包含 submodule**，编译时报：

```
fatal error: cute/tensor.hpp: 没有那个文件或目录
```

**解决**：单独下载 CUTLASS 并放到 `csrc/cutlass/`。

### 坑 2：CUTLASS 版本必须精确匹配

BSA 锁定的 submodule commit 是：

```
a75b4ac483166189a45290783cb0a18af5ff0ea5
```

一开始用最新的 v4.8.1 会报：

```
error: static assertion failed with "TiledMMA requires independent permutations of MNK."
error: class "cute::TiledMMA<...>" has no member "TiledShape_MNK"
```

**解决**：按锁定的 commit 下载，而不是用 release tag。

```bash
curl -sL -o cutlass.zip \
  https://codeload.github.com/NVIDIA/cutlass/zip/a75b4ac483166189a45290783cb0a18af5ff0ea5
```

### 坑 3：并行编译会被 OOM killer 杀掉

`MAX_JOBS=2` 时 nvcc 在编译 `flash_bwd_block_hdim64_bf16_sm80.o` 时被杀：

```
FAILED: [code=255] .../flash_bwd_block_hdim64_bf16_sm80.o
```

**解决**：必须 `MAX_JOBS=1`。单进程峰值内存约 17 GB（28 GB 内存勉强够）。

代价是编译时间翻倍到约 70 分钟。

### 坑 4：验证时必须先 import torch

```bash
python -c "import block_sparse_attn"        # ✗ ImportError: libc10.so
python -c "import torch; import block_sparse_attn"   # ✓
```

`libc10.so` 来自 torch，必须先进程加载。

---

## 完整步骤

```bash
# 1. 取源码
cd /tmp
curl -sL -o bsa.zip https://codeload.github.com/mit-han-lab/Block-Sparse-Attention/zip/refs/heads/main
python -c "import zipfile;zipfile.ZipFile('/tmp/bsa.zip').extractall('/tmp/bsax')"
cd /tmp/bsax/Block-Sparse-Attention-main

# 2. 补 CUTLASS（用锁定的 commit）
cd /tmp
curl -sL -o cl.zip https://codeload.github.com/NVIDIA/cutlass/zip/a75b4ac483166189a45290783cb0a18af5ff0ea5
python -c "import zipfile;zipfile.ZipFile('/tmp/cl.zip').extractall('/tmp/cl')"
CL=$(ls -d /tmp/cl/*/ | head -1)

rm -rf /tmp/bsax/Block-Sparse-Attention-main/csrc/cutlass
mkdir -p /tmp/bsax/Block-Sparse-Attention-main/csrc/cutlass
cp -r "$CL"include /tmp/bsax/Block-Sparse-Attention-main/csrc/cutlass/include

# 3. 单进程编译
cd /tmp/bsax/Block-Sparse-Attention-main
export MAX_JOBS=1
export TORCH_CUDA_ARCH_LIST="8.0"
python -m pip install --no-build-isolation --no-deps .

# 4. 验证
python -c "import torch; import block_sparse_attn; print('BSA OK')"
```

也可以用 `scripts/build_bsa.sh`（已封装以上步骤，带日志）。

---

## 编译成功后的日志特征

FlashVSR 运行时应该看到：

```
[FlashVSR Native+] attention kernel: block_sparse
```

如果是 `using SDPA fallback` 或 `block_sparse_attn unavailable`，说明没装上。

---

## 其他环境参考

官方说明：Block-Sparse-Attention 在 **A100 / A800（Ampere）** 上验证过，
H200 能跑但加速有限，RTX 40/50 系列未验证。

A10 属于 Ampere（sm86），实测可编译、可运行。

CUDA 12.x 环境可以尝试社区 wheel（`hf-mirror.com/smthem/AI-wheel`），
但注意那里的包都是 Windows 版。
