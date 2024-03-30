from abc import *
import torch
import torch.nn as nn
import torch.nn.functional as F

class NetworkBase(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        super(NetworkBase, self).__init__()
    @abstractmethod
    def forward(self, x):
        return x
    
class Network(NetworkBase):
    def __init__(self, layer_num, input_dim, output_dim, hidden_dim, activation_function = torch.relu,last_activation = None):
        super(Network, self).__init__()
        self.activation = activation_function
        self.last_activation = last_activation
        layers_unit = [input_dim]+ [hidden_dim]*(layer_num-1) 
        layers = ([nn.Linear(layers_unit[idx],layers_unit[idx+1]) for idx in range(len(layers_unit)-1)])
        self.layers = nn.ModuleList(layers)
        self.last_layer = nn.Linear(layers_unit[-1],output_dim)
        self.network_init()
    def forward(self, x):
        return self._forward(x)
    def _forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        x = self.last_layer(x)
        if self.last_activation != None:
            x = self.last_activation(x)
        return x
    def network_init(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                layer.bias.data.zero_()

class Actor(Network):
    def __init__(self, layer_num, input_dim, output_dim, hidden_dim, activation_function = torch.tanh,last_activation = None, trainable_std = False):
        super(Actor, self).__init__(layer_num, input_dim, output_dim, hidden_dim, activation_function ,last_activation)
        self.trainable_std = trainable_std
        if self.trainable_std == True:
            self.logstd = nn.Parameter(torch.zeros(1, output_dim))
    def forward(self, x):
        mu = self._forward(x)
        if self.trainable_std == True:
            std = torch.exp(self.logstd)
        else:
            logstd = torch.zeros_like(mu)
            std = torch.exp(logstd)
        return mu,std

class Critic(Network):
    def __init__(self, layer_num, input_dim, output_dim, hidden_dim, activation_function, last_activation = None):
        super(Critic, self).__init__(layer_num, input_dim, output_dim, hidden_dim, activation_function ,last_activation)
        
    def forward(self, *x):
        x = torch.cat(x,-1)
        return self._forward(x)