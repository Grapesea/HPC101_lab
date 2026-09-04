import os
from pathlib import Path

import torch
import torch_npu
import triton
import triton.language as tl

torch.npu.config.allow_internal_format = False
torch.npu.set_device(0)


@triton.jit
def _inspect_kernel(xp, rp, wp, yp, op, eps):
    flat = tl.arange(0, 8192)
    off = tl.program_id(0) * 8192 + flat
    rf = (tl.load(xp + off).to(tl.float32)
          + tl.load(rp + off).to(tl.float32))
    tl.store(op + off, rf.to(tl.float16))
    r = tl.reshape(rf, (8, 1024), can_reorder=True)
    inv = tl.rsqrt(tl.sum(r * r, axis=1) * 0.0009765625 + eps)
    c = tl.arange(0, 1024)
    w = tl.load(wp + c).to(tl.float32)
    y = (r * inv[:, None]) * w[None, :]
    tl.store(yp + off, tl.reshape(y, (8192,)).to(tl.float16))


x = torch.randn((256, 1024), dtype=torch.float16).npu()
r = torch.randn((256, 1024), dtype=torch.float16).npu()
w = torch.randn((1024,), dtype=torch.float16).npu()
y, o = torch.empty_like(x), torch.empty_like(x)
_inspect_kernel[(32,)](x, r, w, y, o, 1e-6,
                       num_stages=2, multibuffer=True)
torch.npu.synchronize()

roots = [Path(os.environ.get("TRITON_CACHE_DIR", "/tmp/tcache")),
         Path(os.environ.get("TRITON_DUMP_DIR", "/tmp/tdump"))]
files = []
for root in roots:
    if root.exists():
        files.extend(p for p in root.rglob("*") if p.is_file())
for p in files:
    print("ARTIFACT", p, p.stat().st_size)

needles = ("reduce", "barrier", "alloc", "local", "load", "store",
           "copy", "buffer", "pipe", "sqrt", "cast")
shown = 0
for p in files:
    if p.suffix.lower() in (".o", ".so", ".bin", ".cubin"):
        continue
    try:
        text = p.read_text(errors="replace")
    except Exception:
        continue
    hits = [line for line in text.splitlines()
            if any(word in line.lower() for word in needles)]
    if not hits:
        continue
    print("IRFILE", p)
    for line in hits[:100]:
        print(line[:500])
        shown += 1
        if shown >= 350:
            raise SystemExit
