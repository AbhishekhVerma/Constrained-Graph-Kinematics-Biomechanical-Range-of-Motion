"""
Alias / Wrapper for `deployment/live_demo.py` (Real-Time PC, MacBook, Jetson & iPhone Telemetry Demo).

Provides full CLI and programmatic backward-compatibility so any invocation of
`python deployment/inference_pc_iphone.py` or `import deployment.inference_pc_iphone`
delegates directly to `deployment.live_demo`.
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deployment.live_demo import (
    ReplayDatasetStreamer,
    build_arg_parser,
    load_or_calibrate_demo_models,
    main,
    render_dashboard_figure,
    run_live_demo,
)

__all__ = [
    "ReplayDatasetStreamer",
    "load_or_calibrate_demo_models",
    "render_dashboard_figure",
    "run_live_demo",
    "build_arg_parser",
    "main",
]

if __name__ == "__main__":
    main()
