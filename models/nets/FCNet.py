import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
import numpy as np


class FCNet(nn.Module):
    """
    Simple Fully-Connected Network (FCN) for Deep Q Learning Network (DQN)
    """
    def __init__(self, num_layers, hidden_units_per_layer, state_size, num_actions,
                 loss_fn, activation_fn, learning_rate, beta1, beta2, weight_decay, device):
        # General housekeeping
        super().__init__()
        self.name = 'DQN_FCN'
        self.epoch = 0
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

        # Stored history
        self.streaming_weight_history = {}
        self.streaming_gradient_history = {}

        self.histogram_weight_history = {}
        self.histogram_gradient_history = {}

        self.gnorm_history = {}
        self.gnorm_total_history = []
        self.wnorm_history = {}
        self.wnorm_total_history = []

        self.layer_list = []
        self.layer_list_names = []

        self.loss = []  # List of loss per step
        self.losses = []  # List of avg losses per epoch

        self.Q = []  # List of Q's per step
        self.Qs = []  # List of avg Q's per epoch

        self.norm_num = 2
        self.bins = 20  # Choice of bins=20 seems to look nice. Subject to change.

        self.init_layer_list()
        self.init_history()
        self.update_hist_list()

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

    def init_layer_list(self):
        """Initializes list of layers for tracking history"""
        nn_module_ignore_list = {'batchnorm', 'activation', 'loss', 'Noise', 'CustomCatGANLayer'}  # List of nn.modules to ignore when constructing layer_list
        self.layer_list = [x for x in self._modules.values() if not any(excl in str(type(x)) for excl in nn_module_ignore_list)]
        self.layer_list_names = [x for x in self._modules.keys() if not any(excl in str(type(self._modules[x])) for excl in nn_module_ignore_list)]

    def init_history(self):
        """Initializes objects for storing history based on layer_list"""
        for layer in self.layer_list:
            self.streaming_weight_history[layer] = {'weight': [], 'bias': []}
            self.streaming_gradient_history[layer] = {'weight': [], 'bias': []}

            self.histogram_weight_history[layer] = {'weight': [], 'bias': []}
            self.histogram_gradient_history[layer] = {'weight': [], 'bias': []}

            self.wnorm_history[layer] = {'weight': [], 'bias': []}
            self.gnorm_history[layer] = {'weight': [], 'bias': []}

    def next_epoch(self):
        """Resets internal storage of training history to stream next epoch"""
        self.epoch += 1

        self.losses.append(np.mean(self.loss))
        self.loss = []

        self.Qs.append(np.mean(self.Q))
        self.Q = []

        self.update_wnormz()
        self.update_gnormz()
        self.update_hist_list()

        for layer in self.layer_list:
            self.streaming_weight_history[layer] = {'weight': [], 'bias': []}
            self.streaming_gradient_history[layer] = {'weight': [], 'bias': []}

    def store_weight_and_grad_norms(self):
        """
        Appends training history for summarization and visualization later. Scales each norm by the number of elements.
        Should be ran once per step per subnet.
        """
        for layer in self.layer_list:
            self.streaming_weight_history[layer]['weight'].append(layer.weight.norm(self.norm_num).detach().cpu().numpy().take(0) / layer.weight.numel())
            self.streaming_weight_history[layer]['bias'].append(layer.bias.norm(self.norm_num).detach().cpu().numpy().take(0) / layer.bias.numel())

            self.streaming_gradient_history[layer]['weight'].append(layer.weight.grad.norm(self.norm_num).detach().cpu().numpy().take(0) / layer.weight.grad.numel())
            self.streaming_gradient_history[layer]['bias'].append(layer.bias.grad.norm(self.norm_num).detach().cpu().numpy().take(0) / layer.bias.grad.numel())

    def update_hist_list(self):
        """
        Updates the histogram history based on the weights at the end of an epoch.
        Should be ran once per epoch per subnet.
        """
        for layer in self.layer_list:
            self.histogram_weight_history[layer]['weight'].append(np.histogram(layer.weight.detach().cpu().numpy().reshape(-1), bins=self.bins))
            self.histogram_weight_history[layer]['bias'].append(np.histogram(layer.bias.detach().cpu().numpy().reshape(-1), bins=self.bins))

            if self.epoch == 0:  # Model is untrained; no gradients exist yet
                self.histogram_gradient_history[layer]['weight'].append(None)
                self.histogram_gradient_history[layer]['bias'].append(None)
            else:
                self.histogram_gradient_history[layer]['weight'].append(np.histogram(layer.weight.grad.detach().cpu().numpy().reshape(-1), bins=self.bins))
                self.histogram_gradient_history[layer]['bias'].append(np.histogram(layer.bias.grad.detach().cpu().numpy().reshape(-1), bins=self.bins))

    def update_wnormz(self):
        """
        Tracks history of desired norm of weights.
        Should be ran once per epoch per subnet.
        :param norm_num: 1 = l1 norm, 2 = l2 norm
        :return: list of norms of weights by layer, as well as overall weight norm
        """
        total_norm = 0
        for layer in self.wnorm_history:
            w_norm = np.linalg.norm(self.streaming_weight_history[layer]['weight'], self.norm_num)
            b_norm = np.linalg.norm(self.streaming_weight_history[layer]['bias'], self.norm_num)
            self.wnorm_history[layer]['weight'].append(w_norm)
            self.wnorm_history[layer]['bias'].append(b_norm)

            if self.norm_num == 1:
                total_norm += abs(w_norm) + abs(b_norm)
            else:
                total_norm += w_norm ** self.norm_num + b_norm ** self.norm_num

        total_norm = total_norm ** (1. / self.norm_num)
        self.wnorm_total_history.append(total_norm)

    def update_gnormz(self):
        """
        Calculates gradient norms by layer as well as overall. Scales each norm by the number of elements.
        Should be ran once per epoch per subnet.
        :param norm_num: 1 = l1 norm, 2 = l2 norm
        :return: list of gradient norms by layer, as well as overall gradient norm
        """
        total_norm = 0
        for layer in self.gnorm_history:
            w_norm = np.linalg.norm(self.streaming_gradient_history[layer]['weight'], self.norm_num) / len(self.streaming_gradient_history[layer]['weight'])
            b_norm = np.linalg.norm(self.streaming_gradient_history[layer]['bias'], self.norm_num) / len(self.streaming_gradient_history[layer]['bias'])

            self.gnorm_history[layer]['weight'].append(w_norm)
            self.gnorm_history[layer]['bias'].append(b_norm)
            if self.norm_num == 1:
                total_norm += abs(w_norm) + abs(b_norm)
            else:
                total_norm += w_norm**self.norm_num + b_norm**self.norm_num
        total_norm = total_norm**(1./self.norm_num) / len(self.gnorm_history)
        self.gnorm_total_history.append(total_norm)

