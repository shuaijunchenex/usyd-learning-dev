import sys

from ml_utils.config_loader import ConfigLoader

sys.path.insert(0, "../ml_fed_learn")
sys.path.insert(0, "../ml_fed_learn/fed_node") 

sys.path.append("..")

print("\n".join(sys.path))

from ml_fed_learn.fed_node.ml_fed_node_client import FedNodeClient
from ml_fed_learn.fed_node.ml_fed_node_server import FedNodeServer

def load_config():
    node_yaml_file = './test_data/node_config_template.yaml'
    yaml_config = ConfigLoader.load(node_yaml_file)
    return yaml_config

def test_node_client():
    yaml_config = load_config()
    print("Loaded YAML data:")
    print(yaml_config)

    client_node = FedNodeClient("client.1", yaml_config)
    client_node.run()
    return

def test_node_server():
    yaml_config = load_config()
    print("Loaded YAML data:")
    print(yaml_config)

    server_node = FedNodeServer(yaml_config)
    server_node.run()
    return


def main():
    test_node_client()
    # test_node_server()


if __name__ == "__main__":
    main()
