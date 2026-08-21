##################################################################
##
## Run-only helpers for the AMSS-NCKU lab driver.
##
## Both run_ABE and run_TwoPunctureABE invoke the executable via the
## shell with stdin redirected from /dev/null and stdout/stderr streamed
## through tee, so output is visible in the terminal and saved to the
## per-run log file. mpirun therefore owns its file descriptors directly;
## the parent Python just waits for it to fully exit (subprocess.call).
## Earlier code used Popen with stdout=PIPE and iterated the pipe, which
## under OpenMPI 5 caused the iterator to return after the first few
## timesteps with the ABE ranks orphaned.
##
##################################################################

import os
import shlex
import subprocess

import AMSS_NCKU_Input as input_data


def _run_and_tee(cmd, log):
    """Run a shell command and stream output to both terminal and log."""
    quoted_log = shlex.quote(log)
    wrapped = f"set -o pipefail; {cmd} 2>&1 | tee {quoted_log}"
    rc = subprocess.call(["bash", "-c", wrapped])
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def _mpiexec_argv():
    """Return an MPI launcher compatible with the configured hybrid layout.

    The OJ supplies ``--bind-to core --map-by slot:pe=5``.  That requests
    12*5 physical cores, while its 60 CPUs are 30 cores/60 hardware threads,
    so OpenMPI exits with mapper status 213 before ABE starts.  Keep only the
    configured launcher executable and construct the validated mapping here;
    this protects direct AMSS_NCKU_Program.py runs as well as run.sh runs.
    """
    configured = shlex.split(os.environ.get("AMSS_MPIEXEC", "mpiexec"))
    launcher = configured[0] if configured else "mpiexec"
    if input_data.OMP_threads > 1:
        return [launcher, "--allow-run-as-root", "--use-hwthread-cpus",
                "--bind-to", "hwthread", "--map-by",
                f"slot:PE={input_data.OMP_threads}"]
    return configured


def _env_tokens():
    """Build env VAR=VALUE tokens forwarded to each MPI rank.

    Uses `mpiexec -n N env VAR=... exe` form, which works on both OpenMPI
    and MPICH (the -x flag was OpenMPI-only)."""
    forced = {}
    if input_data.OMP_threads > 1:
        forced = {
            "OMP_NUM_THREADS": str(input_data.OMP_threads),
            "OMP_DYNAMIC": "FALSE",
            "OMP_PROC_BIND": "close",
            "OMP_PLACES": "threads",
            "OMP_STACKSIZE": "1G",
            "AMSS_DISTRIBUTION_NODES": "30",
        }

    tokens = []
    for var in ("OMP_NUM_THREADS", "OMP_DYNAMIC", "OMP_PROC_BIND", "OMP_PLACES",
                "OMP_STACKSIZE",
                "OMP_WAIT_POLICY", "GOMP_SPINCOUNT", "OPENBLAS_NUM_THREADS",
                "LD_LIBRARY_PATH", "AMSS_PERF_PREFIX",
                "AMSS_DISTRIBUTION_NODES"):
        value = forced.get(var, os.environ.get(var))
        if value is not None:
            tokens.append(f"{var}={value}")
    return tokens


def _wrapped_executable(exe):
    """Return an optional per-rank wrapper followed by the executable.

    Normal runs do not set ``AMSS_ABE_WRAPPER`` and are unchanged.  A wrapper
    can inspect MPI rank variables and attach perf to one rank only while the
    other ranks execute the binary directly.
    """
    wrapper = os.environ.get("AMSS_ABE_WRAPPER")
    if not wrapper:
        return [exe]
    return shlex.split(wrapper) + [exe]


def run_TwoPunctureABE():
    """Run the TwoPuncture initial-data solver in the current directory."""
    print("\n Running the AMSS-NCKU executable file TwoPunctureABE\n")
    cmd = "./TwoPunctureABE < /dev/null"
    _run_and_tee(cmd, "TwoPunctureABE_out.log")
    print("\n The TwoPunctureABE simulation is finished\n")


def run_ABE():
    """Run the main ABE evolution executable via mpiexec."""
    if input_data.GPU_Calculation == "no":
        exe, log = "./ABE", "ABE_out.log"
    elif input_data.GPU_Calculation == "yes":
        exe, log = "./ABEGPU", "ABEGPU_out.log"
    else:
        raise ValueError("GPU_Calculation must be 'no' or 'yes'")

    print(f"\n Running {exe} with {input_data.MPI_processes} MPI ranks "
          f"and OMP_NUM_THREADS={input_data.OMP_threads}\n")

    # Build the launcher command. `env VAR=... exe` form works on both
    # OpenMPI and MPICH; the original -x flag was OpenMPI-only.
    argv = (_mpiexec_argv()
            + ["-n", str(input_data.MPI_processes)]
            + ["env"] + _env_tokens()
            + _wrapped_executable(exe))
    print(" MPI launcher: " + shlex.join(argv[:-1]))
    cmd = shlex.join(argv) + " < /dev/null"

    _run_and_tee(cmd, log)

    print(f"\n The {exe} simulation is finished\n")
