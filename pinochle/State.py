import numpy as np
from util.Vectors import Vectors as vs
from util.util import print_divider
import logging
from config import Config as cfg
import torch
import util.vector_builder as vb

logging.basicConfig(format='%(levelname)s:%(message)s', level=cfg.logging_level)
"""
The strategy is to create a global state with all available information.
Players will only see the portion of the global state that is available to them.
48 cards in the deck (2 each of 24 different cards). First 24 values will correspond to 
a single player's hand, values of 0-2. Next 24 values to the second player's hand.
"""


class State:
    def __init__(self, game, played_card):
        self.game = game
        self.one_hot_template = vs.PINOCHLE_ONE_HOT_VECTOR

        self.scores_vector = np.array((self.game.scores[self.game.players[0]][-1], self.game.scores[self.game.players[1]][-1]))
        self.player1_hand_vector = vb.build_hand_vector(self.game.hands[self.game.players[0]])
        self.player2_hand_vector = vb.build_hand_vector(self.game.hands[self.game.players[1]])
        self.discard_vector = vb.build_hand_vector(self.game.discard_pile)
        self.trump_vector = vb.build_trump_vector(self.game.trump)
        self.played_card_vector = vb.build_card_vector(played_card)

        """
        ALWAYS HAVE PLAYER 1 AND PLAYER 2 HAND AT THE BEGINNING OF THE STATE
        """
        self.global_state = np.concatenate((self.scores_vector, self.player1_hand_vector, self.player2_hand_vector, self.trump_vector, self.discard_vector, self.played_card_vector), axis=0)

    def convert_to_human_readable_format(self, player):
        print_divider()
        logging.debug(f'Trump Suit: {self.game.trump}')

        print_divider()
        self.game.hands[player].show()

        print_divider()
        self.game.melds[player].show()

        current_scores = [self.game.scores[player][-1] for player in self.game.players]
        print_divider()
        logging.debug("Player 1 Score: " + str(current_scores[0]))
        logging.debug("Player 2 Score: " + str(current_scores[1]))

    def get_player_state(self, player):
        """
        Return a subset of the global state corresponding to what the player should be able to see
        :param player:
        :return:
        """
        player_index = self.game.players.index(player)

        score_info = np.array((self.global_state[player_index] - self.global_state[1 - player_index])).reshape(-1)
        player_hand = self.global_state[2 + player_index * len(self.one_hot_template):2 + (player_index + 1) * len(self.one_hot_template)]
        global_info = self.global_state[2 + 2 * len(self.one_hot_template):]

        return np.concatenate((player_hand, score_info, global_info), axis=0)

    def get_player_state_as_tensor(self, player, device=None):
        arr = self.get_player_state(player)
        device = cfg.DQN_params['device'] if device is None else device
        return torch.from_numpy(arr).type(torch.float32).to(device)

    def get_valid_action_mask(self, player, is_trick):
        """
        Returns a boolean tensor mask that only allows valid action indices
        Assumes first num_action indices are the player's hand
        :param player: Player of interest
        :param is_trick: Whether this is a hand or meld action (boolean, True if hand, False if meld)
        :return: Boolean Tensor
        """
        if is_trick:
            return self.get_player_state_as_tensor(player)[:cfg.num_actions] > 0
        else:
            # TODO
            # go reverse: for each possible combination check if hand has cards for it
            # apply list_to_dict magic on hand for O(1)
            player
            combinations_vector = vs.MELD_COMBINATIONS_ONE_HOT_VECTOR

            for combo in combinations_vector:



            pass