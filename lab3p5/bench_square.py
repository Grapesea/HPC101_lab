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
def _square_kernel(xp, rp, wp, yp, op, eps, MODE: tl.constexpr):
    flat = tl.arange(0, 8192)
    off = tl.program_id(0) * 8192 + flat
    xh = tl.load(xp + off)
    rh = tl.load(rp + off)
    rf = xh.to(tl.float32) + rh.to(tl.float32)
    tl.store(op + off, rf.to(tl.float16))
    r = tl.reshape(rf, (8, 1024), can_reorder=True)

    if MODE == 0:
        ss = tl.sum(r * r, axis=1)
    elif MODE == 1:
        r16 = r.to(tl.float16)
        ss = tl.sum(r16 * r16, axis=1)
    elif MODE == 2:
        r16 = r.to(tl.float16)
        sq = (r16 * r16).to(tl.float32)
        ss = tl.sum(sq, axis=1)
    elif MODE == 3:
        ss = tl.sum(r * r, axis=1, keep_dims=True)
    elif MODE == 4:
        # Round only the value entering the norm; retain rf for the final output.
        nr = tl.reshape(xh + rh, (8, 1024), can_reorder=True)
        ss = tl.sum(nr * nr, axis=1)
    elif MODE == 5:
        ss = tl.sum(r * r, axis=1)
    elif MODE == 6:
        ss = tl.sum(r * r, axis=1)
    else:
        r16 = r.to(tl.float16)
        ss = tl.sum(r16 * r16, axis=1, keep_dims=True)

    if MODE == 3 or MODE == 7:
        inv2 = tl.rsqrt(ss * 0.0009765625 + eps)
    elif MODE == 5:
        inv = 1.0 / tl.sqrt(ss * 0.0009765625 + eps)
    elif MODE == 6:
        inv = tl.rsqrt(ss * 0.0009765625 + eps).to(tl.float16).to(tl.float32)
    else:
        inv = tl.rsqrt(ss * 0.0009765625 + eps)

    c = tl.arange(0, 1024)
    w = tl.load(wp + c).to(tl.float32)
    if MODE == 3 or MODE == 7:
        y = (r * inv2) * w[None, :]
    else:
        y = (r * inv[:, None]) * w[None, :]
    tl.store(yp + off, tl.reshape(y, (8192,)).to(tl.float16))


def make_inputs(seed, lo=-1.0, hi=1.0):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    x = (torch.rand(256, 1024, generator=g) * (hi - lo) + lo).half()
    r = (torch.rand(256, 1024, generator=g) * (hi - lo) + lo).half()
    w = (torch.rand(1024, generator=g) * 2).clamp(min=.01).half()
    return x, r, w


def child(mode, stages):
    x, r, w = make_inputs(27109)
    rr = x.float() + r.float()
    gy = ((rr / torch.sqrt(torch.mean(rr * rr, dim=1, keepdim=True) + 1e-6))
          * w.float()).half()
    go = rr.half()
    xd, rd, wd = x.npu(), r.npu(), w.npu()
    y, o = torch.empty_like(xd), torch.empty_like(xd)
    if stages == 2:
        _square_kernel[(32,)](xd, rd, wd, y, o, 1e-6, MODE=mode,
                              num_stages=2, multibuffer=True)
    else:
        _square_kernel[(32,)](xd, rd, wd, y, o, 1e-6, MODE=mode,
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
    child(int(sys.argv[2]), int(sys.argv[3]))
    raise SystemExit


specs = [
    ("base-default-A", 0, 1),
    ("base-stage2-A", 0, 2),
    ("sqrt-div-A", 5, 1),
    ("base-default-B", 0, 1),
    ("rounded-inv-A", 6, 1),
    ("base-stage2-B", 0, 2),
    ("sqrt-div-B", 5, 1),
    ("rounded-inv-B", 6, 1),
    ("base-default-C", 0, 1),
]
for name, mode, stages in specs:
    out = "/tmp/farn_focus_" + name
    shutil.rmtree(out, ignore_errors=True)
    print("===== FOCUS", name, mode, stages, "=====", flush=True)
    cmd = ["msprof", "op", "--warm-up=20", "--kernel-name=_square_kernel",
           "--launch-count=100", "--output=" + out,
           sys.executable, __file__, "child", str(mode), str(stages)]
    q = subprocess.run(cmd, text=True, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, timeout=160)
    for line in q.stdout.splitlines():
        if ("Task Duration" in line or line.startswith("cfg ") or
                "Traceback" in line or "Error" in line or "error" in line):
            print(line, flush=True)
