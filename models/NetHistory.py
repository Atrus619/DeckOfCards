import numpy as np
from config import Config as cfg
import os
import pickle as pkl


class NetHistory:
    def __init__(self, _modules):
        self._modules = _modules
        self.epoch = 0

        self.streaming_weight_history = {}
        self.streaming_gradient_history = {}

        self.histogram_weight_history = {}
        self.histogram_gradient_history = {}

        self.gnorm_history = {}
        self.gnorm_total_history = []
        self.wnorm_history = {}
        self.wnorm_total_history = []

        self.loss = []  # List of loss per step
        self.losses = []  # List of avg losses per epoch

        self.Q = []  # List of Q's per step
        self.Qs = []  # List of avg Q's per epoch

        self.norm_num = 2
        self.bins = 20  # Choice of bins=20 seems to look nice. Subject to change.

        self.layer_zip = self.init_layer_zip()  # List of tuples, first index is the actual layer, second index is the layer name
        self.init_history()
        self.update_hist_list()

    def next_epoch(self):
        self.epoch += 1

        self.losses.append(np.mean(self.loss))
        self.loss = []

        self.Qs.append(np.mean(self.Q))
        self.Q = []

        self.update_wnormz()
        self.update_gnormz()
        self.update_hist_list()

        for layer, layer_name in self.layer_zip:
            self.streaming_weight_history[layer_name] = {'weight': [], 'bias': []}
            self.streaming_gradient_history[layer_name] = {'weight': [], 'bias': []}

    def init_layer_zip(self):
        """Initializes list of layers for tracking history"""
        nn_module_ignore_list = {'batchnorm', 'activation', 'loss', 'Noise', 'CustomCatGANLayer'}  # List of nn.modules to ignore when constructing layer_list
        layer_list = [x for x in self._modules.values() if not any(excl in str(type(x)) for excl in nn_module_ignore_list)]
        layer_list_names = [x for x in self._modules.keys() if not any(excl in str(type(self._modules[x])) for excl in nn_module_ignore_list)]
        return list(zip(layer_list, layer_list_names))

    def init_history(self):
        """Initializes objects for storing history based on layer_list"""
        for _, layer_name in self.layer_zip:
            self.streaming_weight_history[layer_name] = {'weight': [], 'bias': []}
            self.streaming_gradient_history[layer_name] = {'weight': [], 'bias': []}

            self.histogram_weight_history[layer_name] = {'weight': [], 'bias': []}
            self.histogram_gradient_history[layer_name] = {'weight': [], 'bias': []}

            self.wnorm_history[layer_name] = {'weight': [], 'bias': []}
            self.gnorm_history[layer_name] = {'weight': [], 'bias': []}

    def update_hist_list(self):
        """
        Updates the histogram history based on the weights at the end of an epoch.
        Should be ran once per epoch per subnet.
        """
        for layer, layer_name in self.layer_zip:
            self.histogram_weight_history[layer_name]['weight'].append(np.histogram(layer.weight.detach().cpu().numpy().reshape(-1), bins=self.bins))
            self.histogram_weight_history[layer_name]['bias'].append(np.histogram(layer.bias.detach().cpu().numpy().reshape(-1), bins=self.bins))

            if len(self.histogram_gradient_history[layer_name]['weight']) == 0:  # Model is untrained; no gradients exist yet
                self.histogram_gradient_history[layer_name]['weight'].append(None)
                self.histogram_gradient_history[layer_name]['bias'].append(None)
            else:
                self.histogram_gradient_history[layer_name]['weight'].append(np.histogram(layer.weight.grad.detach().cpu().numpy().reshape(-1), bins=self.bins))
                self.histogram_gradient_history[layer_name]['bias'].append(np.histogram(layer.bias.grad.detach().cpu().numpy().reshape(-1), bins=self.bins))

    def store_weight_and_grad_norms(self):
        """
        Appends training history for summarization and visualization later. Scales each norm by the number of elements.
        Should be ran once per step per subnet.
        """
        for layer, layer_name in self.layer_zip:
            self.streaming_weight_history[layer_name]['weight'].append(layer.weight.norm(self.norm_num).detach().cpu().numpy().take(0) / layer.weight.numel())
            self.streaming_weight_history[layer_name]['bias'].append(layer.bias.norm(self.norm_num).detach().cpu().numpy().take(0) / layer.bias.numel())

            self.streaming_gradient_history[layer_name]['weight'].append(layer.weight.grad.norm(self.norm_num).detach().cpu().numpy().take(0) / layer.weight.grad.numel())
            self.streaming_gradient_history[layer_name]['bias'].append(layer.bias.grad.norm(self.norm_num).detach().cpu().numpy().take(0) / layer.bias.grad.numel())

    def update_wnormz(self):
        """
        Tracks history of desired norm of weights.
        Should be ran once per epoch per subnet.
        :param norm_num: 1 = l1 norm, 2 = l2 norm
        :return: list of norms of weights by layer, as well as overall weight norm
        """
        total_norm = 0
        for _, layer_name in self.layer_zip:
            w_norm = np.linalg.norm(self.streaming_weight_history[layer_name]['weight'], self.norm_num)
            b_norm = np.linalg.norm(self.streaming_weight_history[layer_name]['bias'], self.norm_num)
            self.wnorm_history[layer_name]['weight'].append(w_norm)
            self.wnorm_history[layer_name]['bias'].append(b_norm)

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
        for _, layer_name in self.layer_zip:
            w_norm = np.linalg.norm(self.streaming_gradient_history[layer_name]['weight'], self.norm_num) / len(self.streaming_gradient_history[layer_name]['weight'])
            b_norm = np.linalg.norm(self.streaming_gradient_history[layer_name]['bias'], self.norm_num) / len(self.streaming_gradient_history[layer_name]['bias'])

            self.gnorm_history[layer_name]['weight'].append(w_norm)
            self.gnorm_history[layer_name]['bias'].append(b_norm)
            if self.norm_num == 1:
                total_norm += abs(w_norm) + abs(b_norm)
            else:
                total_norm += w_norm ** self.norm_num + b_norm ** self.norm_num
        total_norm = total_norm ** (1. / self.norm_num) / len(self.gnorm_history)
        self.gnorm_total_history.append(total_norm)

    def save(self, title, folder=cfg.history_folder):
        # Save itself to disk. If prior history exists, loads this in and appends itself to the old history being saving it to disk (maintains a cumulative history).
        os.makedirs(folder, exist_ok=True)
        full_title = title + '_history.pkl'
        if full_title in os.listdir(folder):  # Prior history object exists
            with open(os.path.join(folder, full_title), 'rb') as f:
                old_self = pkl.load(f)

            new_self = self.custom_append(old_self=old_self)

            with open(os.path.join(folder, full_title), 'wb') as f:
                pkl.dump(new_self, f)

        else:
            with open(os.path.join(folder, full_title), 'wb') as f:
                pkl.dump(self, f)

    def custom_append(self, old_self):
        # Appends current history to old self. Used in saving itself to disk.
        old_self.epoch += self.epoch

        old_self.losses += self.losses
        old_self.Qs += self.Qs

        for _, layer_name in self.layer_zip:
            old_self.wnorm_history[layer_name]['weight'] += self.wnorm_history[layer_name]['weight']
            old_self.wnorm_history[layer_name]['bias'] += self.wnorm_history[layer_name]['bias']

            old_self.wnorm_total_history += self.wnorm_total_history

            old_self.gnorm_history[layer_name]['weight'] += self.gnorm_history[layer_name]['weight']
            old_self.gnorm_history[layer_name]['bias'] += self.gnorm_history[layer_name]['bias']

            old_self.gnorm_total_history += self.gnorm_total_history

        return old_self
