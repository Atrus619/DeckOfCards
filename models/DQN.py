import torch.nn as nn
import torch.nn.functional as F
import torch
from models.nets.FCNet import FCNet
from copy import deepcopy
import pickle as pkl
import os
from config import Config as cfg
import logging
from util.Constants import Constants as cs
from util.Vectors import Vectors as vs

logging.basicConfig(format='%(levelname)s:%(message)s', level=cfg.logging_level)


class DQN:
    """
    Deep Q Learning Network (DQN)
    """
    def __init__(self, update_target_net_freq, gamma, grad_clamp, terminal_state_tensor,
                 num_layers, hidden_units_per_layer, state_size, num_actions,
                 loss_fn=F.smooth_l1_loss, activation_fn=nn.LeakyReLU(0.2), learning_rate=2e-4, beta1=0.5, beta2=0.999, weight_decay=0, device=None):
        # General housekeeping
        self.device = device if device is not None else torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        self.update_target_net_freq = update_target_net_freq
        self.gamma = gamma
        self.grad_clamp = grad_clamp
        self.terminal_state_tensor = terminal_state_tensor.to(self.device)
        self.player = None

        # Instantiate neural nets
        self.policy_net = FCNet(num_layers=num_layers, hidden_units_per_layer=hidden_units_per_layer, state_size=state_size, num_actions=num_actions, loss_fn=loss_fn,
                                activation_fn=activation_fn, learning_rate=learning_rate, beta1=beta1, beta2=beta2, weight_decay=weight_decay, device=device).to(self.device)
        self.target_net = deepcopy(self.policy_net)

        # History
        self.loss = []

    def get_legal_action(self, state, game, player, is_hand):
        # Picks the highest output legal action, ignoring illegal actions
        valid_action_mask = state.get_valid_action_mask(player=player, is_hand=is_hand)
        invalid_action_mask = (valid_action_mask == 0).nonzero()
        initial_action_tensor = self.policy_net(state.get_player_state_as_tensor(player=player))

        # Setting to large negative number to deal with ties.
        # If an invalid option and a valid option had the same value, it could choose the invalid option, leading to a downstream error.
        initial_action_tensor[invalid_action_mask] = -999

        if game.human_test:
            logging.debug(cs.DIVIDER)
            logging.debug("Action values for each card in agent's hand:")
            actions = torch.nn.Softmax(dim=0)(initial_action_tensor)
            for i, action in enumerate(actions):
                card = player.convert_model_output(output_index=i, game=game, is_hand=True)
                if card is None:
                    continue
                logging.debug("Card: " + str(vs.PINOCHLE_ONE_HOT_VECTOR[i]) + "/" + \
                              card + "\t Action value: {0:.2%}".format(action.item()))

        best_valid_action_prob = initial_action_tensor[valid_action_mask.nonzero()].max()

        return (initial_action_tensor == best_valid_action_prob).nonzero()[0].item()

    def train_one_batch(self, states, actions, next_states, rewards, store_history):
        action_indices = actions.argmax(dim=1).unsqueeze(1)
        pred_Q = self.policy_net(states=states).gather(1, action_indices)  # Compute Q(s_t)

        pred_next_state_values = self.target_net(next_states).max(dim=1)[0]  # Compute V(s_t+1) for all next states
        terminal_state_mask = (next_states == self.terminal_state_tensor).all(dim=1)  # Determine which states are terminal
        pred_next_state_values[terminal_state_mask] = 0.0  # Overwrite terminal states with zeros
        expected_Q = (pred_next_state_values * self.gamma) + rewards

        loss = self.policy_net.loss_fn(pred_Q, expected_Q.unsqueeze(1))  # Compute loss

        # Optimize Model
        self.policy_net.zero_grad()
        loss.backward()

        if self.grad_clamp:
            for param in self.policy_net.parameters():
                param.grad.data.clamp_(-1, 1)

        self.policy_net.opt.step()

        # Save History
        if store_history:
            self.policy_net.loss.append(loss.item())
            self.policy_net.Q.append(pred_Q.mean().item())
            self.policy_net.store_weight_and_grad_norms()

    def train_self(self, num_epochs, exp_gen, store_history=False):
        device_mismatch = self.device != exp_gen.dataset.device

        self.policy_net.train()

        for i in range(num_epochs):
            if i % self.update_target_net_freq == 0:
                self.update_target_net()

            for states, actions, next_states, rewards in exp_gen:
                if device_mismatch:
                    states, actions, next_states, rewards = states.to(self.device), actions.to(self.device), next_states.to(self.device), rewards.to(self.device)

                self.train_one_batch(states=states, actions=actions, next_states=next_states, rewards=rewards, store_history=store_history)

            if store_history:
                self.policy_net.next_epoch()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def copy(self):
        return deepcopy(self)

    def save(self, folder=cfg.saved_models_folder, title=None):
        title = 'latest' if title is None else title
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, title + '.pkl'), 'wb') as f:
            pkl.dump(self, f)
