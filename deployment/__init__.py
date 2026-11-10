"""
Edge benchmarking and dual-mode live 3D hand telemetry demo package.

Module Owners: Kapish & Vrushank
"""

from deployment.benchmark_edge import (
     count_parameters,
    compute_hardware_calibrated_profiles,
    detect_hardware_and_backends,
    estimate_architecture_flops,
    format_markdown_report,
    load_or_export_student_onnx_session,
    measure_callable_latency,
    run_all_benchmarks,
)
from deployment.live_demo import (
    ReplayDatasetStreamer,
    load_or_calibrate_demo_models,
    render_dashboard_figure,
    run_live_demo,
)

__all__ = [
    "detect_hardware_and_backends",
    "count_parameters",
    "estimate_architecture_flops",
    "measure_callable_latency",
    "load_or_export_student_onnx_session",
    "compute_hardware_calibrated_profiles",
    "run_all_benchmarks",
    "format_markdown_report",
    "ReplayDatasetStreamer",
    "load_or_calibrate_demo_models",
    "render_dashboard_figure",
    "run_live_demo",
]
