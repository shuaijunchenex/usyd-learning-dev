#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import copy
from typing import Dict, Any

try:
    import yaml  # PyYAML
except ImportError:
    print("请先安装 PyYAML: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

STANDARD_PATHS: Dict[str, str] = {
    "yaml_folder_aggreation_path": "../../yamls/aggregation/",
    "yaml_folder_client_selection_path": "../../yamls/client_selection/",
    "yaml_folder_dataset_loader_path": "../../yamls/dataset_loader/",
    "yaml_folder_data_distribution_path": "../../yamls/data_distribution/",
    "yaml_folder_loss_func_path": "../../yamls/loss_func/",
    "yaml_folder_lora_inference_path": "../../yamls/lora_inference/",
    "yaml_folder_nn_model_path": "../../yamls/nn_model/",
    "yaml_folder_fl_nodes_path": "../../yamls/fl_nodes/",
    "yaml_folder_optimizer_path": "../../yamls/optimizer/",
    "yaml_folder_client_strategy_path": "../../yamls/client_strategy/",
    "yaml_folder_server_strategy_path": "../../yamls/server_strategy/",
    "yaml_folder_runner_strategy_path": "../../yamls/runner_strategy/",
    "yaml_folder_trainer_path": "../../yamls/trainer/",
    "yaml_folder_training_logger_path": "../../yamls/training_logger/",
    "yaml_folder_training_path": "../../yamls/training/",
    "yaml_folder_general_path": "../../yamls/general",
    "yaml_folder_rank_distribution_path": "../../yamls/rank_distribution",
}

def update_paths(cfg: Dict[str, Any]) -> bool:
    changed = False
    for k, v in STANDARD_PATHS.items():
        if k in cfg and cfg.get(k) != v:
            cfg[k] = v
            changed = True
    return changed

def _replace_in_list(items, replace_map):
    changed = False
    new_items = []
    for it in items:
        if isinstance(it, str):
            new_it = it
            for src, tgt in replace_map.items():
                new_it = new_it.replace(src, tgt)
            if new_it != it:
                changed = True
            new_items.append(new_it)
        else:
            new_items.append(it)
    return changed, new_items

def update_dataset_and_distribution(cfg: Dict[str, Any], from_ds: str, to_ds: str) -> bool:
    """
    在 yaml_combination 下的 client_yaml / server_yaml 中，同时替换：
      - dataset_loader_<from>  -> dataset_loader_<to>
      - <from>_one_label       -> <to>_one_label
      - <from>                 -> <to>  （兜底，避免漏网）
    """
    comb = cfg.get("yaml_combination")
    if not isinstance(comb, dict):
        return False

    replace_map = {
        f"dataset_loader_{from_ds}": f"dataset_loader_{to_ds}",
        f"{from_ds}_one_label": f"{to_ds}_one_label",
        from_ds: to_ds,  # 兜底
    }

    changed = False
    for section in ("client_yaml", "server_yaml"):
        if section in comb and isinstance(comb[section], list):
            c, new_list = _replace_in_list(comb[section], replace_map)
            if c:
                comb[section] = new_list
                changed = True
    return changed

# >>> 新增：确保 data_distribution 清单里有 <to_ds>_one_label <<<
def ensure_data_distribution_entry(cfg: Dict[str, Any], to_ds: str) -> bool:
    """
    确保在 yaml_folder_data_distribution_files 列表中存在：
      - "<to_ds>_one_label.yaml" : <to_ds>_one_label
    YAML 里这一节通常是 list[dict]，每个 dict 只有一条 {filename: alias}
    """
    key = "yaml_folder_data_distribution_files"
    files = cfg.get(key)
    if not isinstance(files, list):
        return False

    filename = f"{to_ds}_one_label.yaml"
    alias = f"{to_ds}_one_label"

    # 已存在就不添加
    for item in files:
        if isinstance(item, dict) and filename in item:
            return False

    files.append({filename: alias})
    return True
# <<< 新增结束 <<<

def process_yaml_file(path: str, in_place: bool, backup: bool, from_ds: str, to_ds: str) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        print(f"[SKIP] 读取失败：{path} ({e})")
        return False
    if not isinstance(cfg, dict):
        return False

    original = copy.deepcopy(cfg)
    c1 = update_paths(cfg)
    c2 = update_dataset_and_distribution(cfg, from_ds=from_ds, to_ds=to_ds)
    c3 = ensure_data_distribution_entry(cfg, to_ds=to_ds)  # <<< 调用新增函数
    changed = c1 or c2 or c3
    if not changed:
        return False

    if in_place:
        if backup:
            bak = path + ".bak"
            try:
                with open(bak, "w", encoding="utf-8") as fbak:
                    yaml.dump(original, fbak, sort_keys=False, allow_unicode=True)
            except Exception as e:
                print(f"[WARN] 备份失败：{bak} ({e})")
        try:
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(cfg, f, sort_keys=False, allow_unicode=True)
            print(f"[OK] 写回：{path}")
            return True
        except Exception as e:
            print(f"[ERR] 写回失败：{path} ({e})")
            return False
    else:
        print(f"[DRY] {path} 将被更新（未写回）。")
        return True

def is_yaml(fn: str) -> bool:
    fnl = fn.lower()
    return fnl.endswith(".yaml") or fnl.endswith(".yml")

def run(root=".", from_dataset="fmnist", to_dataset="kmnist",
        recursive=True, backup=True, in_place=True):
    if not os.path.isdir(root):
        print(f"[ERR] 目录不存在：{root}")
        return 1

    scanned = changed = 0
    walker = os.walk(root) if recursive else [(root, [], os.listdir(root))]
    for dirpath, _, filenames in walker:
        for fn in filenames:
            if not is_yaml(fn):
                continue
            p = os.path.join(dirpath, fn)
            scanned += 1
            if process_yaml_file(
                p,
                in_place=in_place,
                backup=backup,
                from_ds=from_dataset,
                to_ds=to_dataset,
            ):
                changed += 1

    print(f"\n扫描 YAML：{scanned} 个，更新写回：{changed} 个。（from={from_dataset} -> to={to_dataset}）")
    return 0

def main(argv=None):
    ap = argparse.ArgumentParser(description="批量更新 YAML：统一路径；同时修改数据集与分布")
    ap.add_argument("--root", default=".", help="包含 YAML 的目录（默认当前目录）")
    ap.add_argument("--from-dataset", default="fmnist", help="源数据集名（默认 fmnist）")
    ap.add_argument("--to-dataset", default="kmnist", help="目标数据集名（默认 kmnist）")
    ap.add_argument("--in-place", action="store_true", default=True, help="原地写回（默认开）")
    ap.add_argument("--no-in-place", dest="in_place", action="store_false", help="只预览不写回")
    ap.add_argument("--backup", action="store_true", default=False, help="写回前生成 .bak 备份（默认关/按你当前设置）")
    ap.add_argument("--no-backup", dest="backup", action="store_false", help="不生成备份")
    ap.add_argument("--recursive", action="store_true", default=True, help="递归处理子目录（默认开）")
    ap.add_argument("--no-recursive", dest="recursive", action="store_false", help="不递归")

    args = ap.parse_args(argv)

    return run(
        root= r"C:\MyPhD\usyd-learning-src\src\test\fl_lora_sample\convergence_experiment\qmnist",
        from_dataset="kmnist",
        to_dataset="qmnist",
        recursive=args.recursive,
        backup=False,
        in_place=args.in_place,
    )

if __name__ == "__main__":
    raise SystemExit(main())
