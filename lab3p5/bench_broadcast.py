#!/usr/local/python3.11.14/bin/python3
import shutil
import subprocess
import sys

import torch
import torch_npu
import triton
import triton.language as tl

torch.npu.config.allow_internal_format = False
torch.npu.set_device(0)


@triton.jit
def _broadcast_kernel(xp, rp, wp, yp, op, eps, MODE: tl.constexpr):
    pid = tl.program_id(0)
    flat = tl.arange(0, 8192)
    off = pid * 8192 + flat
    rf = (tl.load(xp + off).to(tl.float32)
          + tl.load(rp + off).to(tl.float32))
    tl.store(op + off, rf.to(tl.float16))

    r = tl.reshape(rf, (8, 1024), can_reorder=True)
    inv = tl.rsqrt(tl.sum(r * r, axis=1) * 0.0009765625 + eps)
    cols = tl.arange(0, 1024)

    if MODE == 0:
        w = tl.load(wp + cols).to(tl.float32)
        y = (r * inv[:, None]) * w[None, :]
        yf = tl.reshape(y, (8192,))
    elif MODE == 1:
        # Trade repeated GM reads for removal of the full-tile weight broadcast.
        wf = tl.load(wp + (flat & 1023)).to(tl.float32)
        inv2 = tl.broadcast_to(inv[:, None], (8, 1024))
        yf = rf * tl.reshape(inv2, (8192,)) * wf
    elif MODE == 2:
        # Keep the reduction exact, but narrow only the normalized activation.
        w = tl.load(wp + cols)
        n = (r * inv[:, None]).to(tl.float16)
        y = n * w[None, :]
        yf = tl.reshape(y, (8192,))
    elif MODE == 3:
        wf = tl.load(wp + (flat & 1023))
        inv2 = tl.broadcast_to(inv[:, None], (8, 1024))
        nf = (rf * tl.reshape(inv2, (8192,))).to(tl.float16)
        yf = nf * wf
    elif MODE == 4:
        w = tl.load(wp + cols).to(tl.float32)
        y = (r * w[None, :]) * inv[:, None]
        yf = tl.reshape(y, (8192,))
    elif MODE == 5:
        w = tl.load(wp + cols).to(tl.float32)
        scale = inv[:, None] * w[None, :]
        yf = tl.reshape(r * scale, (8192,))
    elif MODE == 6:
        w = tl.load(wp + cols).to(tl.float32)
        ih = inv.to(tl.float16)
        y = (r * ih[:, None]) * w[None, :]
        yf = tl.reshape(y, (8192,))
    else:
        w = tl.load(wp + cols)
        ih = inv.to(tl.float16)
        y = (r.to(tl.float16) * ih[:, None]) * w[None, :]
        yf = tl.reshape(y, (8192,))

    tl.store(yp + off, yf.to(tl.float16))


def make_inputs(seed, lo=-1.0, hi=1.0):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    x = (torch.rand(256, 1024, generator=g) * (hi - lo) + lo).half()
    r = (torch.rand(256, 1024, generator=g) * (hi - lo) + lo).half()
    w = (torch.rand(1024, generator=g) * 2).clamp(min=.01).half()
    return x, r, w


def child(mode):
    x, r, w = make_inputs(19000 + mode * 37)
    rr = x.float() + r.float()
    gy = ((rr / torch.sqrt(torch.mean(rr * rr, dim=1, keepdim=True) + 1e-6))
          * w.float()).half()
    go = rr.half()
    xd, rd, wd = x.npu(), r.npu(), w.npu()
    y, o = torch.empty_like(xd), torch.empty_like(xd)
    _broadcast_kernel[(32,)](xd, rd, wd, y, o, 1e-6, MODE=mode,
                             multibuffer=True)
    torch.npu.synchronize()
    yy, oo = y.cpu(), o.cpu()
    ay = (yy.float() - gy.float()).abs()
    ao = (oo.float() - go.float()).abs()
    by = int(((ay > 1e-3) &
              (ay / gy.float().abs().clamp_min(1e-12) > 1e-3)).sum())
    bo = int(((ao > 1e-3) &
              (ao / go.float().abs().clamp_min(1e-12) > 1e-3)).sum())
    print("cfg", mode, "bad", by, bo,
          "neq", int((yy != gy).sum()), int((oo != go).sum()), flush=True)


if len(sys.argv) > 1 and sys.argv[1] == "child":
    child(int(sys.argv[2]))
    raise SystemExit


specs = [
    ("baseA", 0),
    ("repeat-w", 1),
    ("norm-f16", 2),
    ("repeat-norm-f16", 3),
    ("weight-first", 4),
    ("scale-first", 5),
    ("inv-f16", 6),
    ("all-f16", 7),
    ("baseB", 0),
]
for name, mode in specs:
    out = "/tmp/farn_broadcast_" + name
    shutil.rmtree(out, ignore_errors=True)
    print("===== BROADCAST", name, mode, "=====", flush=True)
    cmd = ["msprof", "op", "--warm-up=20",
           "--kernel-name=_broadcast_kernel", "--launch-count=100",
           "--output=" + out, sys.executable, __file__, "child", str(mode)]
    q = subprocess.run(cmd, text=True, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, timeout=160)
    for line in q.stdout.splitlines():
        if ("Task Duration" in line or line.startswith("cfg ") or
                "Traceback" in line or "Error" in line or "error" in line):
            print(line, flush=True)
