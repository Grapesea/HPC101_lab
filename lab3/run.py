# 想理解测试在测什么，可以从 main() 依次读到 _run_core_case()
import argparse
import contextlib
import ctypes
import ctypes.util
import os
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


_PROCESS_START = time.perf_counter()
_PROFILE_PHASES = os.environ.get("GDN_PROFILE_PHASES") == "1"
_IMPORT_TIMINGS: list[tuple[str, float]] = []


def _set_cuda_blocking_sync_before_init() -> None:
    """Select blocking host waits before PyTorch creates a CUDA context."""

    if os.environ.get("GDN_CUDA_BLOCKING_SYNC") != "1":
        return
    cudart_name = ctypes.util.find_library("cudart") or "libcudart.so"
    cudart = ctypes.CDLL(cudart_name)
    cuda_device_schedule_blocking_sync = 0x04
    result = cudart.cudaSetDeviceFlags(cuda_device_schedule_blocking_sync)
    if result != 0:
        raise RuntimeError(f"cudaSetDeviceFlags(blocking sync) failed: {result}")


_set_cuda_blocking_sync_before_init()

_import_start = time.perf_counter()
import torch
_IMPORT_TIMINGS.append(("import_torch", time.perf_counter() - _import_start))

_import_start = time.perf_counter()
from evaluation.support import (
    DEFAULT_REPETITIONS,
    DEFAULT_WARMUP,
    Case,
    Inputs,
    assert_close,
    load_cases,
    make_inputs,
)
_IMPORT_TIMINGS.append(("import_evaluation", time.perf_counter() - _import_start))

_import_start = time.perf_counter()
from preprocessing.tilelang_cumsum import chunk_local_cumsum
from preprocessing.tilelang_kkt_solve import kkt_solve
_IMPORT_TIMINGS.append(("import_preprocessing", time.perf_counter() - _import_start))

_import_start = time.perf_counter()
from references.torch_gdr import ref_chunk_gated_delta_rule
_IMPORT_TIMINGS.append(("import_reference", time.perf_counter() - _import_start))

_import_start = time.perf_counter()
from student.tilelang_fwd import gdn_prefill_forward
_IMPORT_TIMINGS.append(("import_student", time.perf_counter() - _import_start))




DEFAULT_CASES = Path(__file__).parent / "evaluation" / "cases.csv"
ANSI_GREEN = "\033[32m"
ANSI_RED = "\033[31m"
ANSI_RESET = "\033[0m"

FullForward = Callable[
    [
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
    ],
    tuple[torch.Tensor, torch.Tensor],
]


@dataclass
class Measurement:
    status: str
    median_ms: float | None = None


@contextlib.contextmanager
def _profile_range(name: str):
    """Add an opt-in NVTX range and a matching host wall-time record."""

    if not _PROFILE_PHASES:
        yield
        return
    start = time.perf_counter()
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()
        elapsed_ms = (time.perf_counter() - start) * 1e3
        print(f"PROFILE_PHASE {name} {elapsed_ms:.3f} ms", file=sys.stderr)


def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # 一个完整 forward 会先准备 g_cumsum 和 A，再进入你实现的 W/U/S/O。
    with _profile_range("full_forward/chunk_local_cumsum"):
        g_cumsum = chunk_local_cumsum(raw_g)
    with _profile_range("full_forward/kkt_solve"):
        A = kkt_solve(k, g_cumsum, beta)
    with _profile_range("full_forward/student_forward"):
        return gdn_prefill_forward(q, k, v, g_cumsum, beta, A, initial_state)


def _call_full_forward(
    implementation: FullForward,
    inputs: Inputs,
) -> tuple[torch.Tensor, torch.Tensor]:
    return implementation(
        inputs.q,
        inputs.k,
        inputs.v,
        inputs.g,
        inputs.beta,
        inputs.initial_state,
    )


def _check_full_forward(
    implementation_name: str,
    implementation: FullForward,
    inputs: Inputs,
    expected_output: torch.Tensor,
    expected_state: torch.Tensor,
) -> None:
    actual_output, actual_state = _call_full_forward(implementation, inputs)
    _check_result(
        implementation_name,
        actual_output,
        actual_state,
        expected_output,
        expected_state,
    )


def _check_result(
    implementation_name: str,
    actual_output: torch.Tensor,
    actual_state: torch.Tensor,
    expected_output: torch.Tensor,
    expected_state: torch.Tensor,
) -> None:
    # output 和跨 chunk 传递的 final_state 都必须正确
    assert actual_output.dtype == torch.bfloat16
    assert actual_state.dtype == torch.float32
    assert_close(f"{implementation_name} output", actual_output, expected_output)
    assert_close(f"{implementation_name} final_state", actual_state, expected_state)


def _benchmark(
    function: Callable[[], tuple[torch.Tensor, torch.Tensor]],
    warmup: int,
    repetitions: int,
) -> float:
    with _profile_range(f"benchmark/warmup_{warmup}"):
        for _ in range(warmup):
            function()
        torch.cuda.synchronize()

    profile_after_warmup = os.environ.get("GDN_PROFILE_AFTER_WARMUP") == "1"
    if profile_after_warmup:
        torch.cuda.cudart().cudaProfilerStart()

    batch_event_sync = os.environ.get("GDN_BATCH_EVENT_SYNC") == "1"
    event_pairs = []
    durations = []
    with _profile_range(f"benchmark/timed_{repetitions}"):
        for _ in range(repetitions):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            function()
            end.record()
            if batch_event_sync:
                event_pairs.append((start, end))
            else:
                end.synchronize()
                durations.append(start.elapsed_time(end))
        if batch_event_sync:
            torch.cuda.synchronize()
            durations.extend(start.elapsed_time(end) for start, end in event_pairs)
    if profile_after_warmup:
        torch.cuda.cudart().cudaProfilerStop()
    return statistics.median(durations)


def _load_full_implementations() -> dict[str, tuple[str, FullForward]]:

    from references.official.fla import chunk_gated_delta_rule as fla
    from references.official.flash_qla import chunk_gated_delta_rule as flash_qla
    from references.official.flashinfer import chunk_gated_delta_rule as flashinfer

    return {
        "student": ("Student", chunk_gated_delta_rule),
        "flash_qla": ("FlashQLA", flash_qla),
        "fla": ("FLA", fla),
        "flashinfer": ("FlashInfer", flashinfer),
    }


def _select_cases(cases: list[Case], requested: list[str] | None) -> list[Case]:
    if not requested:
        return cases
    cases_by_name = {case.name: case for case in cases}
    return [cases_by_name[name] for name in requested]


def _print_table_header(cases: list[Case]) -> int:
    case_width = max(12, *(len(case.name) for case in cases))
    print(
        f"{'CASE':<{case_width}}  {'B':>2}  {'T':>7}  {'Hq':>3}  {'Hv':>3}  "
        f"{'STATE':>5}  {'RESULT':>6}  {'MEDIAN (ms)':>12}"
    )
    print("-" * (case_width + 52))
    return case_width


def _color(text: str, color: str, enabled: bool) -> str:
    return f"{color}{text}{ANSI_RESET}" if enabled else text


def _print_table_row(
    case: Case,
    measurement: Measurement,
    case_width: int,
    *,
    color: bool,
) -> None:
    state = "yes" if case.use_initial_state else "no"
    result = _color(
        f"{measurement.status:>6}",
        ANSI_GREEN if measurement.status == "PASS" else ANSI_RED,
        color,
    )
    latency = (
        f"{measurement.median_ms:>12.3f}"
        if measurement.median_ms is not None
        else f"{'—':>12}"
    )
    print(
        f"{case.name:<{case_width}}  {case.batch_size:>2}  {case.seqlen:>7}  "
        f"{case.num_heads_qk:>3}  {case.num_heads_v:>3}  {state:>5}  "
        f"{result}  {latency}"
    )


def _print_device_summary() -> None:
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    memory_gib = properties.total_memory / 2**30
    print(
        f"Benchmark device: {properties.name} | "
        f"CC {properties.major}.{properties.minor} | "
        f"{properties.multi_processor_count} SMs | {memory_gib:.2f} GiB"
    )


def _print_full_table_header(cases: list[Case]) -> tuple[int, int, int]:
    case_width = max(12, *(len(case.name) for case in cases))
    implementation_width = len("IMPLEMENTATION")
    line_width = case_width + implementation_width + 37
    print(
        f"{'CASE':<{case_width}}  "
        f"{'IMPLEMENTATION':<{implementation_width}}  "
        f"{'RESULT':>6}  {'MEDIAN (ms)':>12}  {'REF/STUDENT':>11}"
    )
    print("-" * line_width)
    return case_width, implementation_width, line_width


def _reference_over_student(
    implementation_name: str,
    measurement: Measurement,
    student_measurement: Measurement,
) -> float | None:
    if implementation_name == "student":
        return None
    if measurement.median_ms is None or student_measurement.median_ms is None:
        return None
    return measurement.median_ms / student_measurement.median_ms


def _print_full_table_row(
    case: Case,
    implementation_name: str,
    implementation_label: str,
    measurement: Measurement,
    student_measurement: Measurement,
    case_width: int,
    implementation_width: int,
    *,
    color: bool,
) -> None:
    result = _color(
        f"{measurement.status:>6}",
        ANSI_GREEN if measurement.status == "PASS" else ANSI_RED,
        color,
    )
    latency = (
        f"{measurement.median_ms:>12.3f}"
        if measurement.median_ms is not None
        else f"{'—':>12}"
    )
    ratio = _reference_over_student(
        implementation_name,
        measurement,
        student_measurement,
    )
    ratio_text = f"{ratio:>10.3f}x" if ratio is not None else f"{'—':>11}"
    print(
        f"{case.name:<{case_width}}  "
        f"{implementation_label:<{implementation_width}}  "
        f"{result}  {latency}  {ratio_text}"
    )

# 核心forward计时
def _run_core_case(
    case: Case,
    warmup: int,
    repetitions: int,
) -> Measurement:
    # initiate inputs
    with _profile_range(f"{case.name}/make_inputs"):
        inputs = make_inputs(case)

    # get correctness reference
    with _profile_range(f"{case.name}/reference_forward"):
        expected_output, expected_state = ref_chunk_gated_delta_rule(
            inputs.q,
            inputs.k,
            inputs.v,
            inputs.g,
            inputs.beta,
            inputs.initial_state,
        )

    # precompute
    with _profile_range(f"{case.name}/preprocess_cumsum"):
        g_cumsum = chunk_local_cumsum(inputs.g)
    with _profile_range(f"{case.name}/preprocess_kkt"):
        A = kkt_solve(inputs.k, g_cumsum, inputs.beta)

    def call_student() -> tuple[torch.Tensor, torch.Tensor]:
        return gdn_prefill_forward(
            inputs.q,
            inputs.k,
            inputs.v,
            g_cumsum,
            inputs.beta,
            A,
            inputs.initial_state,
        )

    # check correctness of your implementations
    try:
        with _profile_range(f"{case.name}/first_student_call_compile"):
            output, state = call_student()
        with _profile_range(f"{case.name}/correctness_check"):
            _check_result("student", output, state, expected_output, expected_state)
    except AssertionError as error:
        print(f"{case.name}: {error}", file=sys.stderr)
        return Measurement(status="FAIL")

    # timing your implementations
    if case.purpose == "correctness":
        return Measurement(status="PASS")
    with _profile_range(f"{case.name}/benchmark"):
        median_ms = _benchmark(call_student, warmup, repetitions)
    return Measurement(status="PASS", median_ms=median_ms)

# 完整forward计时
def _run_full_case(
    case: Case,
    implementations: dict[str, tuple[str, FullForward]],
    warmup: int,
    repetitions: int,
) -> dict[str, Measurement]:

    measurements: dict[str, Measurement] = {}
    inputs = make_inputs(case)
    expected_output, expected_state = ref_chunk_gated_delta_rule(
        inputs.q,
        inputs.k,
        inputs.v,
        inputs.g,
        inputs.beta,
        inputs.initial_state,
    )

    for name, (label, implementation) in implementations.items():
        try:
            _check_full_forward(
                label,
                implementation,
                inputs,
                expected_output,
                expected_state,
            )
        except AssertionError as error:
            print(f"{case.name} [{label}]: {error}", file=sys.stderr)
            measurements[name] = Measurement(status="FAIL")
        else:
            measurements[name] = Measurement(status="PASS")

    if case.purpose == "correctness":
        return measurements

    del expected_output, expected_state
    for name, (_, implementation) in implementations.items():
        if measurements[name].status != "PASS":
            continue
        measurements[name].median_ms = _benchmark(
            lambda implementation=implementation: _call_full_forward(
                implementation, inputs
            ),
            warmup,
            repetitions,
        )
    return measurements


def _format_latency(measurement: Measurement) -> str:
    return f"{measurement.median_ms:.6f}" if measurement.median_ms is not None else ""


def _print_core_csv(
    case: Case,
    measurement: Measurement,
    *,
    extended: bool,
) -> None:
    latency = _format_latency(measurement)
    if extended:
        print(f"{case.name},student_core,student,{measurement.status},{latency},")
    else:
        print(f"{case.name},{measurement.status},{latency}")


def _print_full_csv(
    case: Case,
    implementation_name: str,
    measurement: Measurement,
    student_measurement: Measurement,
) -> None:
    latency = _format_latency(measurement)
    ratio = _reference_over_student(
        implementation_name,
        measurement,
        student_measurement,
    )
    ratio_text = f"{ratio:.6f}" if ratio is not None else ""
    print(
        f"{case.name},full_forward,{implementation_name},"
        f"{measurement.status},{latency},{ratio_text}"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Check complete GDN forward correctness, time student W/U/S/O, "
            "and optionally benchmark official full-forward references."
        )
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument(
        "--case",
        action="append",
        help="Run one named CSV case; may be specified more than once.",
    )
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument(
        "--output-format",
        choices=("table", "csv"),
        default="table",
        help="Output a readable table (default) or machine-readable CSV.",
    )
    parser.add_argument(
        "--reference-benchmarks",
        action="store_true",
        help=(
            "Also time complete student, FlashQLA, FLA, and FlashInfer forwards. "
            "This triggers additional first-run JIT compilation."
        ),
    )
    args = parser.parse_args(argv)

    with _profile_range("main/load_cases"):
        cases = _select_cases(load_cases(args.cases), args.case)

    if args.output_format == "csv":
        if args.reference_benchmarks:
            print(
                "case,scope,implementation,correctness,median_ms,reference_over_student"
            )
        else:
            print("case,correctness,median_ms")
        case_width = 0
    else:
        case_width = _print_table_header(cases)

    use_color = (
        args.output_format == "table"
        and sys.stdout.isatty()
        and "NO_COLOR" not in os.environ
    )
    core_failed = 0
    for case in cases:
        with _profile_range(f"core_case/{case.name}"):
            measurement = _run_core_case(
                case,
                warmup=args.warmup,
                repetitions=args.repetitions,
            )
        if measurement.status != "PASS":
            core_failed += 1
        if args.output_format == "csv":
            _print_core_csv(
                case,
                measurement,
                extended=args.reference_benchmarks,
            )
        else:
            _print_table_row(
                case,
                measurement,
                case_width,
                color=use_color,
            )

    if args.output_format == "table":
        print("-" * (case_width + 52))
        if core_failed:
            summary = f"{core_failed} of {len(cases)} student-core case(s) failed."
            print(_color(summary, ANSI_RED, use_color))
        else:
            summary = f"All {len(cases)} case(s) passed correctness."
            print(_color(summary, ANSI_GREEN, use_color))

    full_failed = 0
    if args.reference_benchmarks:
        implementations = _load_full_implementations()
        if args.output_format == "table":
            print()
            _print_device_summary()
            print("Complete forward references (REF/STUDENT = reference / student):")
            full_case_width, implementation_width, full_line_width = (
                _print_full_table_header(cases)
            )

        for case in cases:
            measurements = _run_full_case(
                case,
                implementations,
                warmup=args.warmup,
                repetitions=args.repetitions,
            )
            student_measurement = measurements["student"]
            for implementation_name, (label, _) in implementations.items():
                measurement = measurements[implementation_name]
                if measurement.status != "PASS":
                    full_failed += 1
                if args.output_format == "csv":
                    _print_full_csv(
                        case,
                        implementation_name,
                        measurement,
                        student_measurement,
                    )
                else:
                    _print_full_table_row(
                        case,
                        implementation_name,
                        label,
                        measurement,
                        student_measurement,
                        full_case_width,
                        implementation_width,
                        color=use_color,
                    )

        if args.output_format == "table":
            print("-" * full_line_width)
            total_rows = len(cases) * len(implementations)
            if full_failed:
                summary = (
                    f"{full_failed} of {total_rows} complete-forward result(s) failed."
                )
                print(_color(summary, ANSI_RED, use_color))
            else:
                summary = f"All {total_rows} complete-forward result(s) passed."
                print(_color(summary, ANSI_GREEN, use_color))

    if core_failed or full_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    if _PROFILE_PHASES:
        for import_name, elapsed in _IMPORT_TIMINGS:
            print(
                f"PROFILE_IMPORT {import_name} {elapsed * 1e3:.3f} ms",
                file=sys.stderr,
            )
        print(
            f"PROFILE_IMPORT startup_to_main "
            f"{(time.perf_counter() - _PROCESS_START) * 1e3:.3f} ms",
            file=sys.stderr,
        )
        with _profile_range("main"):
            main()
    else:
        main()
