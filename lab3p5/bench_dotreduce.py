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
def _dot_kernel(xp, rp, wp, yp, op, eps, MODE: tl.constexpr):
    if MODE == 0:
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
    else:
        lanes = tl.arange(0, 16)
        cols = tl.arange(0, 1024)
        if MODE == 4:
            rows = tl.program_id(0) * 8 + lanes
            mask = lanes[:, None] < 8
        else:
            rows = tl.program_id(0) * 16 + lanes
            mask = lanes[:, None] < 16
        off = rows[:, None] * 1024 + cols[None, :]
        xh = tl.load(xp + off, mask=mask, other=0.0)
        rh = tl.load(rp + off, mask=mask, other=0.0)
        if MODE == 2:
            rf = xh.to(tl.float32) + rh.to(tl.float32)
            rhalf = rf.to(tl.float16)
        else:
            rhalf = xh + rh
        tl.store(op + off, rhalf, mask=mask)
        gram = tl.dot(rhalf, tl.trans(rhalf), out_dtype=tl.float32)
        diag_mask = lanes[:, None] == lanes[None, :]
        diag = tl.sum(tl.where(diag_mask, gram, 0.0), axis=1)
        inv = tl.rsqrt(diag * 0.0009765625 + eps)
        wh = tl.load(wp + cols)
        if MODE == 2:
            y = (rf * inv[:, None]) * wh.to(tl.float32)[None, :]
        elif MODE == 3:
            y = (rhalf * inv.to(tl.float16)[:, None]) * wh[None, :]
        else:
            y = (rhalf.to(tl.float32) * inv[:, None]) * wh.to(tl.float32)[None, :]
        tl.store(yp + off, y.to(tl.float16), mask=mask)


def make_inputs(seed):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    x = (torch.rand(256, 1024, generator=g) * 2 - 1).half()
    r = (torch.rand(256, 1024, generator=g) * 2 - 1).half()
    w = (torch.rand(1024, generator=g) * 2).clamp(min=.01).half()
    return x, r, w


def child(mode):
    x, r, w = make_inputs(18000 + mode * 29)
    R = x.float() + r.float()
    gy = ((R / torch.sqrt(torch.mean(R * R, dim=1, keepdim=True) + 1e-6))
          * w.float()).half()
    go = R.half()
    xd, rd, wd = x.npu(), r.npu(), w.npu()
    y, o = torch.empty_like(xd), torch.empty_like(xd)
    grid = (32,) if mode == 0 or mode == 4 else (16,)
    _dot_kernel[grid](xd, rd, wd, y, o, 1e-6, MODE=mode,
                      multibuffer=True)
    torch.npu.synchronize()
    yy, oo = y.cpu(), o.cpu()
    ay, ao = (yy.float() - gy.float()).abs(), (oo.float() - go.float()).abs()
    by = int(((ay > 1e-3) &
              (ay / gy.float().abs().clamp_min(1e-12) > 1e-3)).sum())
    bo = int(((ao > 1e-3) &
              (ao / go.float().abs().clamp_min(1e-12) > 1e-3)).sum())
    print("cfg", mode, "grid", grid[0], "bad", by, bo,
          "neq", int((yy != gy).sum()), int((oo != go).sum()), flush=True)


if len(sys.argv) > 1 and sys.argv[1] == "child":
    child(int(sys.argv[2]))
    raise SystemExit


specs = [
    ("baseA", 0),
    ("dot16half", 1),
    ("dot16fp32", 2),
    ("dot16allhalf", 3),
    ("dot8half", 4),
    ("dot16halfB", 1),
    ("dot16fp32B", 2),
    ("baseB", 0),
]
for name, mode in specs:
    out = "/tmp/farn_dot_" + name
    shutil.rmtree(out, ignore_errors=True)
    print("===== DOT", name, mode, "=====", flush=True)
    cmd = ["msprof", "op", "--warm-up=20", "--kernel-name=_dot_kernel",
           "--launch-count=100", "--output=" + out,
           sys.executable, __file__, "child", str(mode)]
    q = subprocess.run(cmd, text=True, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, timeout=180)
    for line in q.stdout.splitlines():
        if ("Task Duration" in line or line.startswith("cfg ") or
                "Traceback" in line or "Error" in line or "error" in line or
                "CompilationError" in line):
            print(line, flush=True)
