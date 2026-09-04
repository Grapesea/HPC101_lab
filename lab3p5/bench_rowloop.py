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
def _rowloop_kernel(xp, rp, wp, yp, op, eps,
                    TILE_M: tl.constexpr, LOOP_MODE: tl.constexpr):
    pid = tl.program_id(0)
    cols = tl.arange(0, 1024)
    weight = tl.load(wp + cols).to(tl.float32)
    lanes = tl.arange(0, TILE_M)

    if LOOP_MODE == 0:
        flat = tl.arange(0, 8192)
        off = pid * 8192 + flat
        rf = (tl.load(xp + off).to(tl.float32)
              + tl.load(rp + off).to(tl.float32))
        tl.store(op + off, rf.to(tl.float16))
        r = tl.reshape(rf, (8, 1024), can_reorder=True)
        inv = tl.rsqrt(tl.sum(r * r, axis=1) * 0.0009765625 + eps)
        y = (r * inv[:, None]) * weight[None, :]
        tl.store(yp + off, tl.reshape(y, (8192,)).to(tl.float16))

    elif LOOP_MODE == 1:
        for base in tl.range(0, 8, TILE_M, num_stages=1):
            rows = pid * 8 + base + lanes
            off = rows[:, None] * 1024 + cols[None, :]
            rf = (tl.load(xp + off).to(tl.float32)
                  + tl.load(rp + off).to(tl.float32))
            tl.store(op + off, rf.to(tl.float16))
            inv = tl.rsqrt(tl.sum(rf * rf, axis=1) * 0.0009765625 + eps)
            tl.store(yp + off,
                     ((rf * inv[:, None]) * weight[None, :]).to(tl.float16))

    elif LOOP_MODE == 2:
        for base in tl.range(0, 8, TILE_M, num_stages=2):
            rows = pid * 8 + base + lanes
            off = rows[:, None] * 1024 + cols[None, :]
            rf = (tl.load(xp + off).to(tl.float32)
                  + tl.load(rp + off).to(tl.float32))
            tl.store(op + off, rf.to(tl.float16))
            inv = tl.rsqrt(tl.sum(rf * rf, axis=1) * 0.0009765625 + eps)
            tl.store(yp + off,
                     ((rf * inv[:, None]) * weight[None, :]).to(tl.float16))

    else:
        for base in tl.static_range(0, 8, TILE_M):
            rows = pid * 8 + base + lanes
            off = rows[:, None] * 1024 + cols[None, :]
            rf = (tl.load(xp + off).to(tl.float32)
                  + tl.load(rp + off).to(tl.float32))
            tl.store(op + off, rf.to(tl.float16))
            inv = tl.rsqrt(tl.sum(rf * rf, axis=1) * 0.0009765625 + eps)
            tl.store(yp + off,
                     ((rf * inv[:, None]) * weight[None, :]).to(tl.float16))


def make_inputs(seed):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    x = (torch.rand(256, 1024, generator=g) * 2 - 1).half()
    r = (torch.rand(256, 1024, generator=g) * 2 - 1).half()
    w = (torch.rand(1024, generator=g) * 2).clamp(min=.01).half()
    return x, r, w


def child(tile_m, loop_mode):
    x, r, w = make_inputs(15000 + tile_m * 17 + loop_mode)
    R = x.float() + r.float()
    gy = ((R / torch.sqrt(torch.mean(R * R, dim=1, keepdim=True) + 1e-6))
          * w.float()).half()
    go = R.half()
    xd, rd, wd = x.npu(), r.npu(), w.npu()
    y, o = torch.empty_like(xd), torch.empty_like(xd)
    _rowloop_kernel[(32,)](xd, rd, wd, y, o, 1e-6,
                           TILE_M=tile_m, LOOP_MODE=loop_mode,
                           multibuffer=True)
    torch.npu.synchronize()
    yy, oo = y.cpu(), o.cpu()
    ay, ao = (yy.float() - gy.float()).abs(), (oo.float() - go.float()).abs()
    by = int(((ay > 1e-3) &
              (ay / gy.float().abs().clamp_min(1e-12) > 1e-3)).sum())
    bo = int(((ao > 1e-3) &
              (ao / go.float().abs().clamp_min(1e-12) > 1e-3)).sum())
    print("cfg", tile_m, loop_mode, "bad", by, bo,
          "neq", int((yy != gy).sum()), int((oo != go).sum()), flush=True)


if len(sys.argv) > 1 and sys.argv[1] == "child":
    child(int(sys.argv[2]), int(sys.argv[3]))
    raise SystemExit


specs = [
    ("baseA", 8, 0),
    ("range4s1", 4, 1),
    ("range4s2", 4, 2),
    ("static4", 4, 3),
    ("range2s1", 2, 1),
    ("range2s2", 2, 2),
    ("static2", 2, 3),
    ("range1s2", 1, 2),
    ("range4s2B", 4, 2),
    ("baseB", 8, 0),
]
for name, tile_m, loop_mode in specs:
    out = "/tmp/farn_rowloop_" + name
    shutil.rmtree(out, ignore_errors=True)
    print("===== ROWLOOP", name, tile_m, loop_mode, "=====", flush=True)
    cmd = ["msprof", "op", "--warm-up=20",
           "--kernel-name=_rowloop_kernel", "--launch-count=100",
           "--output=" + out, sys.executable, __file__, "child",
           str(tile_m), str(loop_mode)]
    q = subprocess.run(cmd, text=True, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, timeout=160)
    for line in q.stdout.splitlines():
        if ("Task Duration" in line or line.startswith("cfg ") or
                "Traceback" in line or "Error" in line or "error" in line):
            print(line, flush=True)
