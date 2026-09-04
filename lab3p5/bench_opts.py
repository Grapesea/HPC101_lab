import os, sys, subprocess, shutil
import torch
import torch_npu
import triton
import triton.language as tl

torch.npu.config.allow_internal_format = False
torch.npu.set_device(0)

@triton.jit
def _opt_kernel(xp, rp, wp, yp, op, eps, MODE: tl.constexpr):
    flat = tl.arange(0, 8192)
    off = tl.program_id(0) * 8192 + flat
    if MODE == 3:
        rf = (tl.load(xp + off) + tl.load(rp + off)).to(tl.float32)
    else:
        rf = tl.load(xp + off).to(tl.float32) + tl.load(rp + off).to(tl.float32)
    if MODE == 0:
        tl.store(op + off, rf.to(tl.float16))
    else:
        tl.store(op + off, rf)
    r = tl.reshape(rf, (8, 1024), can_reorder=True)
    if MODE == 6:
        rnorm = r.to(tl.float16).to(tl.float32)
    else:
        rnorm = r
    ms = tl.sum(rnorm * rnorm, axis=1) * 0.0009765625
    inv = tl.rsqrt(ms + eps)
    c = tl.arange(0, 1024)
    w = tl.load(wp + c).to(tl.float32)
    if MODE == 2 or MODE == 3:
        y = r * (inv[:, None] * w[None, :])
    elif MODE == 4:
        scale = (inv[:, None] * w[None, :]).to(tl.float16)
        y = r * scale
    elif MODE == 5:
        y = (r * inv.to(tl.float16)[:, None]) * w[None, :]
    elif MODE == 6:
        y = (rnorm * inv[:, None]) * w[None, :]
    elif MODE == 7:
        norm = (r * inv[:, None]).to(tl.float16)
        y = norm * w[None, :]
    elif MODE == 8:
        invh = inv.to(tl.float16)
        norm = (r * invh[:, None]).to(tl.float16)
        y = norm * w[None, :]
    else:
        y = (r * inv[:, None]) * w[None, :]
    tl.store(yp + off, tl.reshape(y, (8192,)).to(tl.float16))

OPTS = [
    {},
    {"num_stages": 2, "multibuffer": True},
    {"num_stages": 1, "multibuffer": True},
    {"multibuffer": False},
    {"enable_auto_bind_sub_block": False},
    {"limit_auto_multi_buffer_only_for_local_buffer": True},
    {"limit_auto_multi_buffer_only_for_local_buffer": False},
    {"limit_auto_multi_buffer_of_local_buffer": "no-limit"},
    {"limit_auto_multi_buffer_of_local_buffer": "no-l0c"},
    {"enable_hivm_auto_cv_balance": True},
    {"enable_hivm_auto_cv_balance": False},
    {"inject_barrier_all": False},
    {"inject_block_all": False},
    {"tile_mix_vector_loop": 2},
    {"tile_mix_vector_loop": 4},
    {"tile_mix_vector_loop": 8},
    {"inject_barrier_all": False, "num_stages": 2, "multibuffer": True},
    {"enable_hivm_auto_cv_balance": False, "num_stages": 2, "multibuffer": True},
    {"tile_mix_vector_loop": 8, "num_stages": 2, "multibuffer": True},
    {"inject_barrier_all": False, "enable_hivm_auto_cv_balance": False},
    {"inject_barrier_all": False, "tile_mix_vector_loop": 8},
    {"inject_barrier_all": False, "enable_auto_bind_sub_block": False},
    {"inject_barrier_all": False, "enable_fp_fusion": False},
    {"inject_barrier_all": False, "enable_auto_bind_sub_block": False, "num_stages": 2, "multibuffer": True},
    {"inject_barrier_all": False, "enable_auto_bind_sub_block": False, "num_warps": 1},
    {"inject_barrier_all": False, "enable_auto_bind_sub_block": False, "tile_mix_vector_loop": 8},
    {"inject_barrier_all": False, "enable_auto_bind_sub_block": False, "enable_select_analysis": True},
    {"inject_barrier_all": False, "enable_auto_bind_sub_block": False, "enable_hivm_auto_cv_balance": False},
]

def inputs(seed, kind):
    ranges = {"S": (-1., 1.), "M": (1., 10.), "L": (-1000., 1000.)}
    lo, hi = ranges[kind]
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    x = (torch.rand(256, 1024, generator=g) * (hi - lo) + lo).half()
    r = (torch.rand(256, 1024, generator=g) * (hi - lo) + lo).half()
    w = (torch.rand(1024, generator=g) * 2).clamp(min=.01).half()
    return x, r, w

def child(mode, opt):
    x, r, w = inputs(7100 + mode * 31 + opt, "S")
    R = x.float() + r.float()
    gy = ((R / torch.sqrt(torch.mean(R * R, dim=1, keepdim=True) + 1e-6)) * w.float()).half()
    go = R.half()
    xd, rd, wd = x.npu(), r.npu(), w.npu()
    y, o = torch.empty_like(xd), torch.empty_like(xd)
    _opt_kernel[(32,)](xd, rd, wd, y, o, 1e-6, MODE=mode, **OPTS[opt])
    torch.npu.synchronize()
    yy, oo = y.cpu(), o.cpu()
    ay, ao = (yy.float() - gy.float()).abs(), (oo.float() - go.float()).abs()
    by = int(((ay > 1e-3) & (ay / gy.float().abs().clamp_min(1e-12) > 1e-3)).sum())
    bo = int(((ao > 1e-3) & (ao / go.float().abs().clamp_min(1e-12) > 1e-3)).sum())
    print("cfg", mode, opt, "bad", by, bo, "neq", int((yy != gy).sum()), int((oo != go).sum()), flush=True)

if len(sys.argv) > 1 and sys.argv[1] == "child":
    child(int(sys.argv[2]), int(sys.argv[3]))
    raise SystemExit

if len(sys.argv) > 1 and sys.argv[1] == "stress":
    for mode in (0, 5, 7, 8):
        bad_y = bad_o = 0
        max_abs = 0.0
        for kind in ("S", "M", "L"):
            for seed in range(20):
                x, r, w = inputs(9000 + seed, kind)
                R = x.float() + r.float()
                gy = ((R / torch.sqrt(torch.mean(R * R, dim=1, keepdim=True) + 1e-6)) * w.float()).half()
                go = R.half()
                xd, rd, wd = x.npu(), r.npu(), w.npu()
                y, o = torch.empty_like(xd), torch.empty_like(xd)
                _opt_kernel[(32,)](xd, rd, wd, y, o, 1e-6, MODE=mode, num_stages=2, multibuffer=True)
                torch.npu.synchronize()
                yy, oo = y.cpu(), o.cpu()
                ay = (yy.float() - gy.float()).abs()
                ao = (oo.float() - go.float()).abs()
                bad_y += int(((ay > 1e-3) & (ay / gy.float().abs().clamp_min(1e-12) > 1e-3)).sum())
                bad_o += int(((ao > 1e-3) & (ao / go.float().abs().clamp_min(1e-12) > 1e-3)).sum())
                max_abs = max(max_abs, float(ay.max()))
        print("stress", mode, "bad", bad_y, bad_o, "max_abs", max_abs, flush=True)
    raise SystemExit

specs = [
    ("controlA",0,1), ("candidateA",0,21), ("controlB",0,1), ("candidateB",0,21),
    ("bindOnly",0,4), ("candidateStage2",0,23), ("candidateWarp1",0,24),
    ("candidateTile8",0,25), ("candidateSelect",0,26), ("candidateCv",0,27),
    ("candidateC",0,21), ("controlC",0,1),
]
start = int(os.environ.get("BATCH_START", "0"))
end = int(os.environ.get("BATCH_END", str(len(specs))))
for name, mode, opt in specs[start:end]:
    out = "/tmp/farn_opt_" + name
    shutil.rmtree(out, ignore_errors=True)
    print("===== OPT", name, mode, opt, "=====", flush=True)
    cmd = ["msprof", "op", "--warm-up=20", "--kernel-name=_opt_kernel", "--launch-count=100",
           "--output=" + out, sys.executable, __file__, "child", str(mode), str(opt)]
    q = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=160)
    for line in q.stdout.splitlines():
        if "Task Duration" in line or line.startswith("cfg ") or "Traceback" in line or "Error" in line or "unrecognised" in line:
            print(line, flush=True)
