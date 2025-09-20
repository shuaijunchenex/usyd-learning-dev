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

if __name__ == "__main__":

    config_list = list_yaml_files("./fl_lora_sample/convergence_experiment/", "./fl_lora_sample/convergence_experiment/")

    configs = [
        ("./fl_lora_sample/script_test-sp.yaml"),
        ("./fl_lora_sample/script_test-rbla.yaml"),
    ]

    for i in config_list:
        subprocess.run([experiment_entrency.main(i)])
