from __future__ import annotations
import sys

sys.path.append("..")
sys.path.append("../ml_utils")
print("\n".join(sys.path))

from ml_utils.config_loader import ConfigLoader


##############################################

def test_config_loader():
    json_file_name = './test_data/test_config.json'
    yaml_file_name = './test_data/test_config.yaml'
    yaml_file_name_1 = './test_data/test_config.yml'

    # Load JSON file
    json_data = ConfigLoader.load(json_file_name)
    print("Loaded JSON data:")
    print(json_data)

    # Load YAML file
    yaml_data = ConfigLoader.load(yaml_file_name)
    print("Loaded YAML data:")
    print(yaml_data)

    yaml_data = ConfigLoader.load(yaml_file_name_1)
    print("Loaded YAML data:")
    print(yaml_data)

    return

def main():
    test_config_loader()
    return

if __name__ == "__main__":
    main()
