# -*- coding: utf-8 -*-
"""
把 FlashVSR 的 LQ 准备阶段改成「预分配 bf16 + 逐帧填充」。

原实现（plus_sampler / stream_sampler）：
    frames.append(uint8 张量)                 # 1 份 uint8
    lq = torch.stack(frames, 0).float()...    # 再 1 份 uint8 + 1 份 float32
    return lq.to(torch.bfloat16)              # 再 1 份 bf16
  -> 峰值约 3~4 份，552 帧 @2048x1152 约 27 GB

改后：
    lq_buf = torch.empty((3, F, tH, tW), dtype=torch.bfloat16)   # 只 1 份
    每帧：uint8 -> float32(单帧, 很小) -> 归一化 -> 写入 bf16 缓冲
  -> 峰值 1 份 bf16，552 帧约 7.8 GB，数值结果完全一致

用法： python patch_lq_bf16.py <custom_nodes/ComfyUI_FlashVSR 目录>
"""

import os
import shutil
import sys

import py_compile


def patch_plus(path):
    src = open(path, encoding="utf-8").read()

    old = """        frames.append(torch.from_numpy(np.ascontiguousarray(np.asarray(img))).permute(2, 0, 1))
    lq = torch.stack(frames, 0).float().div(127.5).sub(1.0).permute(1, 0, 2, 3).unsqueeze(0)
    return lq.to(torch.bfloat16), tH, tW"""

    new = """        _t = torch.from_numpy(np.ascontiguousarray(np.asarray(img))).permute(2, 0, 1)
        # PATCH(lq_bf16): 预分配 bf16 缓冲，逐帧写入，峰值只占 1 份
        if _lq_buf is None:
            _lq_buf = torch.empty((3, u8.shape[0], tH, tW), dtype=torch.bfloat16)
        _lq_buf[:, i] = _t.float().div_(127.5).sub_(1.0)
    del frames
    return _lq_buf.unsqueeze(0), tH, tW"""

    if old not in src:
        return False, "plus: 目标片段未找到（可能已打过补丁）"
    src = src.replace(old, new, 1)

    # 在循环前插入缓冲初始化
    anchor = "    frames = []\n    for i in range(u8.shape[0]):"
    if anchor not in src:
        return False, "plus: 循环锚点未找到"
    src = src.replace(anchor, "    frames = []\n    _lq_buf = None\n    for i in range(u8.shape[0]):", 1)

    shutil.copyfile(path, path + ".bak_lq_bf16")
    open(path, "w", encoding="utf-8").write(src)
    return True, "plus: 已打补丁"


def patch_stream(path):
    src = open(path, encoding="utf-8").read()

    old = """        arr = torch.from_numpy(np.ascontiguousarray(np.asarray(img)))
        frames.append(arr.permute(2, 0, 1).float().div(127.5).sub(1.0))
    vid = torch.stack(frames, 0).permute(1, 0, 2, 3).unsqueeze(0)  # [1,3,F,tH,tW]
    vid = vid.to(torch.bfloat16)  # reference: pil_to_tensor_neg1_1(dtype=bf16)
    return vid, tH, tW, F, pad_frames"""

    new = """        arr = torch.from_numpy(np.ascontiguousarray(np.asarray(img)))
        # PATCH(lq_bf16): 预分配 bf16 缓冲，逐帧写入，峰值只占 1 份
        if _lq_buf is None:
            _lq_buf = torch.empty((3, len(idx), tH, tW), dtype=torch.bfloat16)
        _lq_buf[:, _pos] = arr.permute(2, 0, 1).float().div_(127.5).sub_(1.0)
        _pos += 1
    del frames
    return _lq_buf.unsqueeze(0), tH, tW, F, pad_frames"""

    if old not in src:
        return False, "stream: 目标片段未找到（可能已打过补丁）"
    src = src.replace(old, new, 1)

    anchor = "    frames = []\n    for i in idx:"
    if anchor not in src:
        return False, "stream: 循环锚点未找到"
    src = src.replace(anchor, "    frames = []\n    _lq_buf = None\n    _pos = 0\n    for i in idx:", 1)

    shutil.copyfile(path, path + ".bak_lq_bf16")
    open(path, "w", encoding="utf-8").write(src)
    return True, "stream: 已打补丁"


def main():
    if len(sys.argv) < 2:
        print("用法: python patch_lq_bf16.py <custom_nodes/ComfyUI_FlashVSR>")
        sys.exit(1)
    root = sys.argv[1]
    nat = os.path.join(root, "flashvsr_native")

    for name, fn in (("plus_sampler.py", patch_plus), ("stream_sampler.py", patch_stream)):
        p = os.path.join(nat, name)
        if not os.path.exists(p):
            print(f"[跳过] {name} 不存在")
            continue
        ok, msg = fn(p)
        print(f"[{'OK' if ok else '!!'}] {msg}")
        if ok:
            py_compile.compile(p, doraise=True)
            print(f"       语法检查通过")


if __name__ == "__main__":
    main()
