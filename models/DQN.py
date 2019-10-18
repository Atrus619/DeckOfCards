import torch.nn as nn
import torch.nn.functional as F
import torch
from models.nets.FCNet import FCNet
from copy import deepcopy


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

    def train_one_batch(self, states, actions, next_states, rewards):
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
        self.loss.append(loss.item())

    def train_self(self, num_epochs, exp_gen):
        device_mismatch = self.device != exp_gen.device

        self.policy_net.train()

        for i in range(num_epochs):
            if i % self.update_target_net_freq == 0:
                self.update_target_net()

            for states, actions, next_states, rewards in exp_gen:
                if device_mismatch:
                    states, actions, next_states, rewards = states.to(self.device), actions.to(self.device), next_states.to(self.device), rewards.to(self.device)

                self.train_one_batch(states=states, actions=actions, next_states=next_states, rewards=rewards)

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def copy(self):
        return deepcopy(self)

    def assign_player(self, player):
        self.player = player

    # TODO: Add history of norms of gradients and weights
