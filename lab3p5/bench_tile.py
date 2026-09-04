import os, sys, subprocess, shutil
import torch
import torch_npu
import triton
import triton.language as tl

torch.npu.config.allow_internal_format = False
torch.npu.set_device(0)

@triton.jit
def _tile_kernel(xp, rp, wp, yp, op, eps,
                 ACTIVE: tl.constexpr, PHYS: tl.constexpr, MODE: tl.constexpr):
    rl = tl.arange(0, PHYS)
    c = tl.arange(0, 1024)
    rows = tl.program_id(0) * ACTIVE + rl
    mask = (rl[:, None] < ACTIVE) & (rows[:, None] < 256)
    off = rows[:, None] * 1024 + c[None, :]
    if MODE == 2:
        w = tl.load(wp + c).to(tl.float32)
    x = tl.load(xp + off, mask=mask, other=0.).to(tl.float32)
    rr = tl.load(rp + off, mask=mask, other=0.).to(tl.float32)
    r = x + rr
    if MODE != 1:
        if MODE == 7:
            tl.store(op + off, r, mask=mask)
        else:
            tl.store(op + off, r.to(tl.float16), mask=mask)
    if MODE == 5:
        ms = tl.sum((r * r) * 0.0009765625, axis=1)
    else:
        ms = tl.sum(r * r, axis=1) * 0.0009765625
    if MODE == 6:
        inv = 1.0 / tl.sqrt(ms + eps)
    else:
        inv = tl.rsqrt(ms + eps)
    if MODE != 2:
        w = tl.load(wp + c).to(tl.float32)
    if MODE == 3:
        y = r * (inv[:, None] * w[None, :])
    elif MODE == 4:
        y = (r * w[None, :]) * inv[:, None]
    else:
        y = (r * inv[:, None]) * w[None, :]
    tl.store(yp + off, y.to(tl.float16), mask=mask)
    if MODE == 1:
        tl.store(op + off, r.to(tl.float16), mask=mask)

def inputs(seed):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    x = (torch.rand(256, 1024, generator=g) * 2 - 1).half()
    r = (torch.rand(256, 1024, generator=g) * 2 - 1).half()
    w = (torch.rand(1024, generator=g) * 2).clamp(min=.01).half()
    return x, r, w

def child(a, p, m):
    x, r, w = inputs(20260904 + a * 19 + m)
    R = x.float() + r.float()
    gy = ((R / torch.sqrt(torch.mean(R * R, dim=1, keepdim=True) + 1e-6)) * w.float()).half()
    go = R.half()
    xd, rd, wd = x.npu(), r.npu(), w.npu()
    y = torch.empty_like(xd)
    o = torch.empty_like(xd)
    grid = ((256 + a - 1) // a,)
    _tile_kernel[grid](xd, rd, wd, y, o, 1e-6, ACTIVE=a, PHYS=p, MODE=m,
                       num_stages=2, multibuffer=True)
    torch.npu.synchronize()
    yy, oo = y.cpu(), o.cpu()
    ay = (yy.float() - gy.float()).abs()
    ao = (oo.float() - go.float()).abs()
    by = int(((ay > 1e-3) & (ay / gy.float().abs().clamp_min(1e-12) > 1e-3)).sum())
    bo = int(((ao > 1e-3) & (ao / go.float().abs().clamp_min(1e-12) > 1e-3)).sum())
    print("cfg", a, p, m, "grid", grid[0], "bad", by, bo, flush=True)

if len(sys.argv) > 1 and sys.argv[1] == "child":
    child(*map(int, sys.argv[2:5]))
    raise SystemExit

specs = [
    ("a4",4,4,0), ("a5",5,8,0), ("a6",6,8,0), ("a7",7,8,0), ("a8",8,8,0),
    ("a9",9,16,0), ("a10",10,16,0), ("a11",11,16,0), ("a12",12,16,0), ("a13",13,16,0),
    ("late",8,8,1), ("wfirst",8,8,2), ("invw",8,8,3), ("rweight",8,8,4),
    ("prescale",8,8,5), ("sqrtdiv",8,8,6), ("autostore",8,8,7),
    ("a7repeat",7,8,0), ("a8repeat",8,8,0),
]
start = int(os.environ.get("BATCH_START", "0"))
end = int(os.environ.get("BATCH_END", str(len(specs))))
for name, a, p, m in specs[start:end]:
    out = "/tmp/farn_tile_" + name
    shutil.rmtree(out, ignore_errors=True)
    print("===== TILE", name, a, p, m, "=====", flush=True)
    cmd = ["msprof", "op", "--warm-up=20", "--kernel-name=_tile_kernel", "--launch-count=100",
           "--output=" + out, sys.executable, __file__, "child", str(a), str(p), str(m)]
    q = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=160)
    for line in q.stdout.splitlines():
        if "Task Duration" in line or line.startswith("cfg ") or "Traceback" in line or "Error" in line:
            print(line, flush=True)
