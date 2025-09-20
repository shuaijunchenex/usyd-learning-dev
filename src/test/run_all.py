import experiment_entrency as experiment_entrency
import subprocess

from pathlib import Path
from typing import List

def list_yaml_files(folder: str, target_path: str) -> List[str]:
    """
    List all .yaml and .yml files under a folder (non-recursive),
    and return paths as target_path + filename.
    
    Args:
        folder (str): Source directory to scan.
        target_path (str): Prefix path for returned file names.
    
    Returns:
        List[str]: List of new paths (target_path + filename).
    """
    folder_path = Path(folder).resolve()
    if not folder_path.is_dir():
        raise NotADirectoryError(f"{folder} is not a valid directory")

    target = Path(target_path)
    files = []
    for p in sorted(list(folder_path.glob("*.yaml")) + list(folder_path.glob("*.yml"))):
        files.append(str(target / p.name))

    return files

# run_all.py
import subprocess
import sys
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def run_all(configs):
    child_code = (
        "import sys; "
        "import experiment_entrency; "
        "experiment_entrency.main(sys.argv[1])"
    )

    env = os.environ.copy()
    sep = ";" if os.name == "nt" else ":"
    env["PYTHONPATH"] = str(BASE_DIR) + (sep + env.get("PYTHONPATH", ""))

    for idx, cfg in enumerate(configs, 1):
        cfg_path = str(Path(cfg).resolve())
        print(f"\n[Batch] ({idx}/{len(configs)}) Running: {cfg_path}")
        cmd = [sys.executable, "-c", child_code, cfg_path]
        subprocess.run(cmd, check=True, cwd=str(BASE_DIR), env=env)
        print(f"[Batch] Finished: {cfg_path}")

if __name__ == "__main__":
    configs = [
        "./fl_lora_sample/script_test-sp.yaml",
        "./fl_lora_sample/script_test-rbla.yaml",
    ]
    config_list = list_yaml_files("./fl_lora_sample/convergence_experiment/", "./fl_lora_sample/convergence_experiment/")
    run_all(config_list)


# if __name__ == "__main__":

#     config_list = list_yaml_files("./fl_lora_sample/convergence_experiment/", "./fl_lora_sample/convergence_experiment/")

#     configs = [
#         ("./fl_lora_sample/script_test-sp.yaml"),
#         ("./fl_lora_sample/script_test-rbla.yaml"),
#     ]

#     for i in config_list:
#         subprocess.run(experiment_entrency.main(i))
