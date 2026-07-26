from centralized import load_data, load_model, train, test

from collections import OrderedDict
import flwr as fl
import torch

def set_parameters(model,parameters):
    param_dict = zip(model.state_dict().keys(),parameters)
    state_dict=OrderedDict({k: torch.tensor(v) for k,v in param_dict})
    model.load_state_dict(state_dict,strict=True)
    return model

net = load_model()
trainloader, testloader = load_data()

class FlowerClient(fl.client.NumPyClient):

    #Returns the current local model weights as a list of NumPy arrays
    def get_parameters(self,config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    #Receives the server's weights, loads them into the local model, trains it on local data
    def fit(self,parameters,config):
        set_parameters(net,parameters)
        train(net, trainloader, epochs=1)
        return self.get_parameters({}), len(trainloader.dataset), {}

    #Receives the server's weights and tests the model on local validation data.
    def evaluate(self,parameters,config):
        set_parameters(net,parameters)
        loss, accuracy = test(net, testloader)
        return float(loss), len(testloader.dataset), {"accuracy": float(accuracy)}

fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=FlowerClient(),
)   
