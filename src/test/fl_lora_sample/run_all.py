# run_many.py
# Usage examples:
#   python run_many.py --dir ./fl_lora_sample/batch --pattern "*.yaml"
#   python run_many.py --configs ./fl_lora_sample/a.yaml ./fl_lora_sample/b.yaml
#   python run_many.py --configs ./fl_lora_sample/a.yaml --stop-on-error

from __future__ import annotations
import argparse
import glob
import os
import shlex
import sys
import textwrap
import subprocess
from pathlib import Path

def discover_configs(base_dir: str, pattern: str) -> list[str]:
    base = Path(base_dir)
    paths = sorted(str(p) for p in base.glob(pattern) if p.is_file())
    return paths

def main():
    parser = argparse.ArgumentParser(description="Run multiple FL experiments in fresh processes.")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--dir", type=str, help="Directory to search for YAMLs.")
    g.add_argument("--configs", nargs="+", help="Explicit list of YAML config files.")
    parser.add_argument("--pattern", type=str, default="*.yaml", help="Glob pattern under --dir (default: *.yaml)")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable to launch children.")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop at first failed experiment.")
    parser.add_argument("--dry-run", action="store_true", help="Only print the commands without running.")
    args = parser.parse_args()

    if args.dir:
        cfgs = discover_configs(args.dir, args.pattern)
    else:
        cfgs = [str(Path(p)) for p in args.configs]

    if not cfgs:
        print("[run_many] No config files found.")
        sys.exit(1)

    # Inline child code: mirror your current entry logic but parameterize config path via argv[1]
    # This runs in a FRESH Python process for each config.
    child_code = textwrap.dedent(r"""
        from __future__ import annotations
        import os, sys
        from usyd_learning.ml_utils import console
        from fl_lora_sample.lora_sample_entry import SampleAppEntry
        from usyd_learning.ml_utils.model_utils import ModelUtils
        from usyd_learning.ml_utils.training_utils import TrainingUtils

        if len(sys.argv) < 2:
            raise SystemExit("Usage: python -c '<code>' <config_yaml_path>")

        config_path = sys.argv[1]

        # ---- mirror your entry's setup ----
        TrainingUtils.set_seed_all(42)

        console.set_log_level("all")
        console.set_debug(True)
        console.set_console_logger(log_path="./log/", log_name="console_trace")
        console.set_exception_logger(log_path="./log/", log_name="exception_trace")
        console.set_debug_logger(log_path="./log/", log_name="debug_trace")
        console.enable_console_log(True)
        console.enable_exception_log(True)
        console.enable_debug_log(True)

        console.out("Simple FL program")
        console.out("======================= PROGRAM BEGIN ==========================")

        g_app = SampleAppEntry()
        g_app.load_app_config(config_path)
        device = ModelUtils.accelerator_device()
        training_rounds = g_app.training_rounds
        g_app.run(device, training_rounds)

        console.out("\\n======================= PROGRAM END ============================")
        # No wait_any_key() in non-interactive batch runs
    """)

    failures = 0
    for idx, cfg in enumerate(cfgs, 1):
        print(f"\n[run_many] ({idx}/{len(cfgs)}) Running: {cfg}")

        cmd = [args.python, "-c", child_code, cfg]

        if args.dry_run:
            print("[run_many] DRY-RUN:", " ".join(shlex.quote(c) for c in cmd))
            continue

        # Launch child as a fresh process
        try:
            completed = subprocess.run(cmd, check=True)
            print(f"[run_many] SUCCESS: {cfg}")
        except subprocess.CalledProcessError as e:
            failures += 1
            print(f"[run_many] FAILED ({cfg}) with return code {e.returncode}")
            if args.stop_on_error:
                sys.exit(e.returncode)

    if failures:
        print(f"\n[run_many] Completed with {failures} failure(s).")
        sys.exit(1)
    else:
        print("\n[run_many] All experiments completed successfully.")

if __name__ == "__main__":
    main()
