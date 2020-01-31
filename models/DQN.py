import torch.nn as nn
import torch.nn.functional as F
import torch
from models.nets.DQN_FCNet import DQN_FCNet
from copy import deepcopy
import pickle as pkl
import os
from config import Config as cfg
import logging
from util.Constants import Constants as cs
from util.Vectors import Vectors as vs
import util.vector_builder as vb

logging.basicConfig(format='%(levelname)s:%(message)s', level=cfg.logging_level)


class DQN:
    """
    Deep Q Learning Network (DQN)
    """

    def __init__(self, update_target_net_freq, gamma, grad_clamp, terminal_state_tensor,
                 num_layers, hidden_units_per_layer, state_size, num_actions,
                 run_id=None, loss_fn=F.smooth_l1_loss, activation_fn=nn.LeakyReLU(0.2), learning_rate=2e-4, beta1=0.5,
                 beta2=0.999, weight_decay=0, device=None):
        # General housekeeping
        self.run_id = 'latest' if run_id is None else run_id
        self.device = device if device is not None else torch.device('cuda:0' if (torch.cuda.is_available()) else 'cpu')
        self.update_target_net_freq = update_target_net_freq
        self.gamma = gamma
        self.grad_clamp = grad_clamp
        self.terminal_state_tensor = terminal_state_tensor.to(self.device)
        self.player = None

        # Instantiate neural nets
        self.policy_net = DQN_FCNet(run_id=self.run_id, num_layers=num_layers,
                                    hidden_units_per_layer=hidden_units_per_layer, state_size=state_size,
                                    num_actions=num_actions,
                                    loss_fn=loss_fn,
                                    activation_fn=activation_fn, learning_rate=learning_rate, beta1=beta1, beta2=beta2,
                                    weight_decay=weight_decay, device=device).to(
            self.device)
        self.target_net = deepcopy(self.policy_net)

        # History
        self.loss = []

        # Configuration
        self.trick_mask_vector, self.meld_mask_vector = torch.tensor(vb.build_trick_mask_vector(),
                                                                     device=self.device), torch.tensor(
            vb.build_meld_mask_vector(), device=self.device)
        self.mask_value = -1

    def get_legal_action(self, state, player, game, is_trick):
        """
        :param state:
        :param player:
        :param game:
        :param is_trick: false represents that the player won the trick thus it will collect the following trick and meld action
        :return:
        """
        # Picks the highest output legal action, ignoring illegal actions
        valid_trick_mask, valid_meld_mask = state.get_valid_action_mask(player=player, is_trick=is_trick)
        initial_trick_action_tensor, initial_meld_action_tensor = \
            self.policy_net(state.get_player_state_as_tensor(player=player))

        if game.human_test:
            self.print_decision_logic(initial_action_tensor=initial_trick_action_tensor, player=player, game=game)

        return self.get_best_masked_action(initial_trick_action_tensor, valid_trick_mask), \
               self.get_best_masked_action(initial_meld_action_tensor, valid_meld_mask)

    @staticmethod
    def get_best_masked_action(tensor, mask):
        # Setting to large negative number to deal with ties.
        # If an invalid option and a valid option had the same value, it could choose the invalid option, leading to a downstream error.
        invalid_mask = (mask == 0).nonzero()
        tensor[invalid_mask] = -999

        best_valid_action_prob = tensor[mask.nonzero()].max()

        return (tensor == best_valid_action_prob).nonzero()[0].item()

    def get_action_indices(self, action_tensor, is_trick):
        # Accepts an action tensor, compares it against the mask tensor (all -1's) and returns the indices of each selected action.
        # Actions that should be masked (no opportunity for a move) are set to -1.
        action_indices = action_tensor.argmax(dim=1).unsqueeze(1)
        action_masks = torch.all(action_tensor == (self.trick_mask_vector if is_trick else self.meld_mask_vector),
                                 dim=1).nonzero().squeeze()
        action_indices[action_masks] = self.mask_value
        return action_indices.squeeze()

    def compute_loss(self, raw_forward_pass_values_current_state, raw_forward_pass_values_next_state, actions,
                     terminal_state_mask, rewards, is_trick):
        # Get indices of each selected action, as well as indicators for masked actions
        raw_action_indices = self.get_action_indices(actions, is_trick=is_trick)
        action_mask = (raw_action_indices != self.mask_value)

        # Filter out invalid / irrelevant actions (i.e. melds that did not occur)
        forward_pass_values_current_state = raw_forward_pass_values_current_state[action_mask, :]
        action_indices = raw_action_indices[action_mask].unsqueeze(1)
        forward_pass_values_next_state = raw_forward_pass_values_next_state[action_mask, :]

        # Gather the predicted Q(s_t) for each selected action
        pred_Q = forward_pass_values_current_state.gather(1, action_indices)

        # Compute V(s_t+1) for the next state
        pred_next_state_best_value = forward_pass_values_next_state.max(dim=1)[0]

        # Determine which next_states are terminal and override with value of 0
        pred_next_state_best_value[terminal_state_mask] = 0.0

        # Calc expected Q
        expected_Q = (pred_next_state_best_value * self.gamma) + rewards[action_mask]
        expected_Q = expected_Q.unsqueeze(1)

        # Calc loss
        loss = self.policy_net.loss_fn(pred_Q, expected_Q)
        return loss

    def train_one_batch(self, states, actions, meld_actions, next_states, rewards, is_storing_history):
        # Forward pass for all states and next_states
        trick_action_outputs_current_state, meld_action_outputs_current_state = self.policy_net(states=states)
        trick_action_outputs_next_state, meld_action_outputs_next_state = self.target_net(states=next_states)
        terminal_state_mask = (next_states == self.terminal_state_tensor).all(
            dim=1)  # Determine which states are terminal

        trick_loss = self.compute_loss(raw_forward_pass_values_current_state=trick_action_outputs_current_state,
                                       raw_forward_pass_values_next_state=trick_action_outputs_next_state,
                                       actions=actions,
                                       terminal_state_mask=terminal_state_mask,
                                       rewards=rewards,
                                       is_trick=True)

        meld_loss = self.compute_loss(raw_forward_pass_values_current_state=meld_action_outputs_current_state,
                                      raw_forward_pass_values_next_state=meld_action_outputs_next_state,
                                      actions=meld_actions,
                                      terminal_state_mask=terminal_state_mask,
                                      rewards=rewards,
                                      is_trick=False)

        # Optimize Model
        self.policy_net.zero_grad()
        total_loss = trick_loss + meld_loss
        total_loss.backward()

        if self.grad_clamp:
            for param in self.policy_net.parameters():
                param.grad.data.clamp_(-1, 1)

        self.policy_net.opt.step()

        # Save History
        if is_storing_history:
            self.policy_net.history.loss.append(total_loss.item())
            # self.policy_net.history.Q.append(pred_Q.mean().item())  # Deprecated for now
            self.policy_net.history.store_weight_and_grad_norms()

    def train_self(self, num_epochs, exp_gen, is_storing_history=False):
        self.policy_net.train()

        for i in range(num_epochs):
            if i % self.update_target_net_freq == 0:
                self.update_target_net()

            for data in exp_gen:
                self.train_one_batch(states=data.state.to(self.device),
                                     actions=data.action.to(self.device),
                                     meld_actions=data.meld_action.to(self.device),
                                     next_states=data.next_state.to(self.device),
                                     rewards=data.reward.to(self.device),
                                     is_storing_history=is_storing_history)

            if is_storing_history:
                self.policy_net.next_epoch()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def copy(self):
        return deepcopy(self)

    def save(self, folder=cfg.final_models_folder, title=None):
        title = self.run_id if title is None else title
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, title + '.pkl'), 'wb') as f:
            pkl.dump(self, f)

    @staticmethod
    def print_decision_logic(initial_action_tensor, player, game):
        logging.debug(cs.DIVIDER)
        logging.debug('Decision Logic For Model:')
        actions = torch.nn.Softmax(dim=0)(initial_action_tensor)

        card_qs = []
        for i, action in enumerate(actions):
            card_qs.append((vs.PINOCHLE_ONE_HOT_VECTOR[i], action.item(),
                            player.convert_model_output(output_index=i, game=game, is_trick=True) is not None))

        card_qs = sorted(card_qs, key=lambda tup: tup[1], reverse=True)
        # card_qs = sorted(card_qs, key=lambda tup: tup[2], reverse=True)

        for card, q, in_hand in card_qs:
            logging.debug('Card: {:20s}Q-Value: {:.2%}\tPresent in hand: {}'.format(str(card), q, in_hand))

    def get_Qs(self, player, player_state, opponent, opponent_state):
        # Returns the Q-value corresponding to the optimal action for both self and opponent. Can be used to determine who it thinks is winning.
        self_Q = self.policy_net(player_state.get_player_state_as_tensor(player=player)).max()
        opponent_Q = self.policy_net(opponent_state.get_player_state_as_tensor(player=opponent)).max()
        score_compare_string = f'Bot Q: {self_Q:.2f} vs. Your Q: {opponent_Q:.2f}'

        if self_Q > opponent_Q:
            return 'The bot thinks it is winning...' + score_compare_string
        elif self_Q < opponent_Q:
            return 'The bot thinks it is losing...' + score_compare_string
        else:
            return 'The bot thinks it is dead even...' + score_compare_string
