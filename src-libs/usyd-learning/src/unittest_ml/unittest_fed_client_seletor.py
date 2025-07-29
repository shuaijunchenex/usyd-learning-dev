from __future__ import annotations


# Init startup path, change current path to test py file folder 
#-----------------------------------------------------------------
import os
from startup_init import startup_init_path
startup_init_path(os.path.dirname(os.path.abspath(__file__)))
#-----------------------------------------------------------------

from usyd_learning.fl_algorithms import FedClientSelectorFactory
from usyd_learning.ml_utils import console, ConfigLoader


##############################################

clients = [i for i in range(10)]
clients_data = {i for i in range(10)}

yaml_file_name = './test_data/node_config_template_server.yaml'

# Load yaml file
console.out(f"Test client selection from {yaml_file_name}")
console.out(f"load yaml file: {yaml_file_name}")
yaml = ConfigLoader.load(yaml_file_name)

def test_client_selector(method):
        
    #Sample 1: simple way
    selector = FedClientSelectorFactory.create(yaml)   #create selector    
    selected_clients = selector.select(clients, 5)            #select client
    print(selected_clients)

    #Sample 2: set random seed
    selector_1 = FedClientSelectorFactory.create(yaml).with_random_seed(142)  #seed
    selected_clients = selector_1.select(clients, 3)
    print(selected_clients)
    return


def test_client_selector_high_loss():
        
    selector = FedClientSelectorFactory.create(yaml).with_clients_data(clients_data)

    selected_client = selector.select(clients, 5)            #select client
    print(selected_client)
    return

def main():
    test_client_selector("random")
    test_client_selector("all")

    test_client_selector_high_loss()
    return

if __name__ == "__main__":
    main()
