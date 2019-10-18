import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
import torch


class FCNet(nn.Module):
    """
    Simple Fully-Connected Network (FCN) for Deep Q Learning Network (DQN)
    """
    def __init__(self, num_layers, hidden_units_per_layer, state_size, num_actions,
                 loss_fn, activation_fn, learning_rate, beta1, beta2, weight_decay, device):
        # General housekeeping
        super().__init__()
        self.device = device

        # RL parameters
        self.state_size = state_size
        self.num_actions = num_actions

        # Layers
        self.num_layers = num_layers
        self.hidden_units_per_layer = hidden_units_per_layer
        self.architecture = OrderedDict()
        self.assemble_architecture()
        self.assemble_modules()

        # Training parameters
        self.act_fn = activation_fn
        self.loss_fn = loss_fn
        self.opt = optim.Adam(self.parameters(), lr=learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)

        # Initialize weights
        self.weights_init()

    def forward(self, states):
        for i, (name, layer) in enumerate(self.architecture.items()):
            if i == 0:
                x = self.act_fn(layer(states))
            elif i < (len(self.architecture) - 1):
                x = self.act_fn(layer(x))
            else:
                return layer(x)  # No activation

    def assemble_architecture(self):
        """
        Assembles architecture of linear layers based on entries in init into an OrderedDict titled self.architecture
        """
        self.architecture['fc_1'] = nn.Linear(in_features=self.state_size, out_features=self.hidden_units_per_layer, bias=True)

        for i in range(1, self.num_layers - 1):
            self.architecture['fc_' + str(i + 1)] = nn.Linear(in_features=self.hidden_units_per_layer, out_features=self.hidden_units_per_layer, bias=True)

        self.architecture['fc_out'] = nn.Linear(in_features=self.hidden_units_per_layer, out_features=self.num_actions, bias=True)

    def assemble_modules(self):
        for name, layer in self.architecture.items():
            self.add_module(name=name, module=layer)

    def weights_init(self):
        """
        Custom weights initialization for subnets
        Should only be run when first creating net. Will reset effects of training if run after training.
        """
        for layer_name in self._modules:
            m = self._modules[layer_name]
            classname = m.__class__.__name__

            if classname.find('Linear') != -1:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
            elif classname.find('BatchNorm') != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
