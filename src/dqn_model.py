import torch.nn
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear

def he_init(m):
   if type(m) == nn.Linear or type(m) == nn.Conv2d:
       torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')

class Network(nn.Module):

    def __init__(self, name):
        super().__init__()
        self.name = name

class FeedForward(Network):

    def __init__(self, input_size, output_size, hidden_layers):
        super().__init__("FeedForward")
        self.output_size = output_size
        hidden_layers = list(input_size) + hidden_layers
        self.layers = nn.ModuleList([Linear(hidden_layers[i],hidden_layers[i+1]) for i in range(len(hidden_layers)-1)])
        self.last_layer = Linear(hidden_layers[-1],output_size)

    def forward(self,x):
        x = x.float()
        for layer in self.layers:
            x = layer(x)
            x = F.relu(x)
        return self.last_layer(x)

class AtariNet(Network):
    def __init__(self, input_shape, n_actions,big=False):
        super(AtariNet, self).__init__("AtariNet" + "Big" if big else "")

        k = 2 if big else 1
        conv_layers = [
        nn.Conv2d(input_shape[0], 16*k, kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(16*k, 32*k, kernel_size=4, stride=2),
        nn.ReLU()]
        if(big):
            conv_layers.append(nn.Conv2d(64, 64, kernel_size=3, stride=1))
            conv_layers.append(nn.ReLU())

        self.conv = nn.Sequential(*conv_layers)

        _,f,w,h = self.conv(torch.zeros([1] + list(input_shape))).shape
        self.conv_out_size = f*w*h
        self.fc = nn.Sequential(
            nn.Linear(self.conv_out_size, 256*k),
            nn.ReLU(),
            nn.Linear(256*k, n_actions)
        )
        self.apply(he_init)

    def forward(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size(0), -1)
        return self.fc(conv_out)

class DuelingAtariNet(AtariNet):
    def __init__(self, input_shape, n_actions,big=False):
        super().__init__(input_shape, n_actions,big)
        self.name += "Dueling"

        k = 2 if big else 1
        self.fc_val = nn.Sequential(
            nn.Linear(self.conv_out_size, 256*k),
            nn.ReLU(),
            nn.Linear(256*k, 1)
        )
        self.apply(he_init)

    def forward(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size(0), -1)
        val = self.fc_val(conv_out)
        adv = self.fc(conv_out)
        return val + adv - adv.mean()

def feed_foward_fn(*hidden_layers):
    return lambda input_size,output_size : FeedForward(input_size,output_size,list(hidden_layers))

def atari_net_fn(big=False):
    return lambda input_size, output_size: AtariNet(input_size, output_size, big)

def atari_net_dueling_fn(big=False):
    return lambda input_size, output_size: DuelingAtariNet(input_size, output_size, big)