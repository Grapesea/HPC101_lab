import os
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
def _dataflow_kernel(xp, rp, wp, yp, op, eps,
                     MODE: tl.constexpr, PART: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * 8 + tl.arange(0, 8)

    if MODE == 0 or MODE == 1:
        flat = tl.arange(0, 8192)
        off = pid * 8192 + flat
        xh = tl.load(xp + off)
        rh = tl.load(rp + off)
        rf = xh.to(tl.float32) + rh.to(tl.float32)
        if MODE == 1:
            tl.store(op + off, xh + rh)
        else:
            tl.store(op + off, rf.to(tl.float16))
        r = tl.reshape(rf, (8, 1024), can_reorder=True)
        inv = tl.rsqrt(tl.sum(r * r, axis=1) * 0.0009765625 + eps)
        c = tl.arange(0, 1024)
        w = tl.load(wp + c).to(tl.float32)
        y = (r * inv[:, None]) * w[None, :]
        tl.store(yp + off, tl.reshape(y, (8192,)).to(tl.float16))

    elif MODE == 2:
        c = tl.arange(0, 512)
        off0 = rows[:, None] * 1024 + c[None, :]
        off1 = off0 + 512
        r0 = tl.load(xp + off0).to(tl.float32) + tl.load(rp + off0).to(tl.float32)
        r1 = tl.load(xp + off1).to(tl.float32) + tl.load(rp + off1).to(tl.float32)
        tl.store(op + off0, r0.to(tl.float16))
        tl.store(op + off1, r1.to(tl.float16))
        ss = tl.sum(r0 * r0, axis=1) + tl.sum(r1 * r1, axis=1)
        inv = tl.rsqrt(ss * 0.0009765625 + eps)
        w0 = tl.load(wp + c).to(tl.float32)
        w1 = tl.load(wp + c + 512).to(tl.float32)
        tl.store(yp + off0, ((r0 * inv[:, None]) * w0[None, :]).to(tl.float16))
        tl.store(yp + off1, ((r1 * inv[:, None]) * w1[None, :]).to(tl.float16))

    else:
        c = tl.arange(0, PART)
        ss = tl.zeros((8,), tl.float32)
        for part in tl.static_range(0, 1024 // PART):
            off = rows[:, None] * 1024 + part * PART + c[None, :]
            rf = (tl.load(xp + off).to(tl.float32)
                  + tl.load(rp + off).to(tl.float32))
            tl.store(op + off, rf.to(tl.float16))
            ss += tl.sum(rf * rf, axis=1)
        inv = tl.rsqrt(ss * 0.0009765625 + eps)
        for part in tl.static_range(0, 1024 // PART):
            off = rows[:, None] * 1024 + part * PART + c[None, :]
            if MODE == 3:
                rf = tl.load(op + off).to(tl.float32)
            else:
                rf = (tl.load(xp + off).to(tl.float32)
                      + tl.load(rp + off).to(tl.float32))
            w = tl.load(wp + part * PART + c).to(tl.float32)
            tl.store(yp + off, ((rf * inv[:, None]) * w[None, :]).to(tl.float16))


def make_inputs(seed):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    x = (torch.rand(256, 1024, generator=g) * 2 - 1).half()
    r = (torch.rand(256, 1024, generator=g) * 2 - 1).half()
    w = (torch.rand(1024, generator=g) * 2).clamp(min=.01).half()
    return x, r, w


def child(mode, part):
    x, r, w = make_inputs(12000 + mode * 31 + part)
    R = x.float() + r.float()
    gy = ((R / torch.sqrt(torch.mean(R * R, dim=1, keepdim=True) + 1e-6))
          * w.float()).half()
    go = R.half()
    xd, rd, wd = x.npu(), r.npu(), w.npu()
    y, o = torch.empty_like(xd), torch.empty_like(xd)
    _dataflow_kernel[(32,)](xd, rd, wd, y, o, 1e-6,
                            MODE=mode, PART=part,
                            num_stages=2, multibuffer=True)
    torch.npu.synchronize()
    yy, oo = y.cpu(), o.cpu()
    ay = (yy.float() - gy.float()).abs()
    ao = (oo.float() - go.float()).abs()
    by = int(((ay > 1e-3) &
              (ay / gy.float().abs().clamp_min(1e-12) > 1e-3)).sum())
    bo = int(((ao > 1e-3) &
              (ao / go.float().abs().clamp_min(1e-12) > 1e-3)).sum())
    print("cfg", mode, part, "bad", by, bo,
          "neq", int((yy != gy).sum()), int((oo != go).sum()), flush=True)


if len(sys.argv) > 1 and sys.argv[1] == "child":
    child(int(sys.argv[2]), int(sys.argv[3]))
    raise SystemExit


specs = [
    ("baseA", 0, 1024),
    ("decoupledA", 1, 1024),
    ("split512A", 2, 512),
    ("reload512", 3, 512),
    ("recompute512", 4, 512),
    ("reload256", 3, 256),
    ("recompute256", 4, 256),
    ("decoupledB", 1, 1024),
    ("split512B", 2, 512),
    ("baseB", 0, 1024),
]
for name, mode, part in specs:
    out = "/tmp/farn_flow_" + name
    shutil.rmtree(out, ignore_errors=True)
    print("===== FLOW", name, mode, part, "=====", flush=True)
    cmd = ["msprof", "op", "--warm-up=20",
           "--kernel-name=_dataflow_kernel", "--launch-count=100",
           "--output=" + out, sys.executable, __file__, "child",
           str(mode), str(part)]
    q = subprocess.run(cmd, text=True, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, timeout=160)
    for line in q.stdout.splitlines():
        if ("Task Duration" in line or line.startswith("cfg ") or
                "Traceback" in line or "Error" in line or "error" in line):
            print(line, flush=True)
