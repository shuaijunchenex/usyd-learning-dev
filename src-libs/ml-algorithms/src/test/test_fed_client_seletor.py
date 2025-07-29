from __future__ import annotations
import sys


sys.path.append("..")
sys.path.append("../fl-algorithms")
print("\n".join(sys.path))

from selection.fed_client_select_method import EFedClientSelectMethod
from selection.fed_client_selector import FedClientSelector


##############################################

clients = [i for i in range(10)]
clients_data = {i for i in range(10)}


def test_client_selector(method: EFedClientSelectMethod):
        
    #Sample 1: simple way
    selector = FedClientSelector.create(clients, method)   #create selector    
    selected_clients = selector.select(5)            #select client
    print(selected_clients)

    #Sample 2: set random seed
    selector_1 = FedClientSelector.create(clients, method).with_random_seed(42)  #seed
    selected_clients = selector_1.select(3)
    print(selected_clients)
    return


def test_client_selector_high_loss():
        
    selector = FedClientSelector \
                .create(clients, EFedClientSelectMethod.high_loss) \
                .with_clients_data(clients_data)

    selected_client = selector.select(5)            #select client
    print(selected_client)
    return

def main():
    test_client_selector(EFedClientSelectMethod.random)
    test_client_selector(EFedClientSelectMethod.all)

    test_client_selector_high_loss()
    return

if __name__ == "__main__":
    main()
