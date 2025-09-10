#!/usr/bin/env python3
"""
PyMetallic Complete Demo and Documentation
Comprehensive demonstration of the PyMetallic library capabilities
"""

import argparse
import os
import sys
import tempfile

from pymetallic import Device, run_simple_compute_example
from pymetallic.helpers import MetallicDemo


def _available_demo_names():
    return sorted(
        attr.removeprefix("demo_")
        for attr in dir(MetallicDemo)
        if attr.startswith("demo_") and callable(getattr(MetallicDemo, attr))
    )


def main():
    """Main entry point"""
    demos_list = _available_demo_names()
    demos_help = (
        "One or more demos to run. Available: " + ", ".join(demos_list) + ". "
        "Use 'all' to run all demos."
    )

    parser = argparse.ArgumentParser(
        description="Run PyMetallic demos",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--quick_test", action="store_true", help="Run a quick device test and exit"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Directory to write demo outputs (images/animations). Defaults to a temporary directory.",
    )
    parser.add_argument(
        "--demos",
        nargs="+",
        choices=demos_list + ["all"],
        default=None,
        metavar="DEMO",
        help=demos_help,
    )

    args = parser.parse_args()

    if args.quick_test:
        print("🔬 PyMetallic Quick Test")
        try:
            device = Device.get_default_device()
            print(f"✅ Metal available: {device.name}")
            run_simple_compute_example()
        except Exception as e:
            print(f"❌ PyMetallic not working: {e}")
        return

    # Resolve and prepare output directory
    out_dir = args.out_dir or tempfile.mkdtemp(prefix="pymetallic-demo-")
    os.makedirs(out_dir, exist_ok=True)

    # Instantiate demo runner with output path
    demo = MetallicDemo(out_path=out_dir)

    # If no demos specified or 'all' selected, run full sequence
    if args.demos is None or "all" in args.demos:
        demo.run_complete_demo()
        return

    # Otherwise, run only selected demos
    available = demo.get_demos()
    unknown = [name for name in args.demos if name not in available]
    if unknown:
        print(f"Unknown demo(s): {', '.join(unknown)}")
        print("Available demos:", ", ".join(sorted(available.keys())))
        sys.exit(2)

    for name in args.demos:
        print(f"🚀 Running {name} demo")
        fn = available[name]
        fn()


if __name__ == "__main__":
    main()
