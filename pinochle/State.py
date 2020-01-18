import numpy as np
from util.Vectors import Vectors as vs
from util.util import print_divider
import logging
from config import Config as cfg
import torch
import util.vector_builder as vb
from pinochle.MeldUtil import MeldUtil

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
        self.player1_meld_vector = vb.build_meld_cards_vector(self.game.melds[self.game.players[0]].melded_cards)
        self.player2_meld_vector = vb.build_meld_cards_vector(self.game.melds[self.game.players[1]].melded_cards)
        self.discard_vector = vb.build_hand_vector(self.game.discard_pile)
        self.trump_vector = vb.build_trump_vector(self.game.trump)
        self.played_card_vector = vb.build_card_vector(played_card)

        self.global_state = np.concatenate((self.scores_vector, self.player1_hand_vector, self.player1_meld_vector,
                                            self.player2_hand_vector, self.player2_meld_vector,
                                            self.trump_vector, self.discard_vector, self.played_card_vector), axis=0)

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
        # Determine if player is player1 (index = 0) or player2 (index = 1)
        player_index = self.game.players.index(player)
        opponent_index = 1 - player_index

        # Array of length 1 describing the differential in score of player vs. opponent (positive means winning, negative means losing)
        score_info = np.array((self.scores_vector[player_index] - self.scores_vector[opponent_index])).reshape(-1)

        # All global information available to both players (excludes melds, these are taken care of in the conditional statement below)
        global_info = np.concatenate((score_info, self.discard_vector, self.trump_vector, self.played_card_vector), axis=0)

        if player_index == 0:  # Player 1
            player_info = np.concatenate((self.player1_hand_vector, self.player1_meld_vector), axis=0)  # Player's hand and meld
            opponent_info = self.player2_meld_vector  # Opponent's meld
        else:  # Player 2
            player_info = np.concatenate((self.player2_hand_vector, self.player2_meld_vector), axis=0)
            opponent_info = self.player1_meld_vector

        # Order of entries in a player's observed state
        return np.concatenate((player_info, opponent_info, global_info), axis=0)

    def get_player_state_as_tensor(self, player, device=None):
        arr = self.get_player_state(player)
        device = cfg.DQN_params['device'] if device is None else device
        return torch.from_numpy(arr).type(torch.float32).to(device)

    def get_valid_action_mask(self, player, is_trick):
        """
        Returns a boolean tensor mask that only allows valid action indices
        Assumes first num_action indices are the player's hand
        :param player: Player of interest
        :param is_trick: false represents that the player won the trick thus it will collect the following trick and meld action
        :return: two boolean tensors
        """
        player_index = self.game.players.index(player)
        if player_index == 0:
            hand_vector = self.player1_hand_vector
            meld_vector = self.player1_meld_vector
        else:
            hand_vector = self.player2_hand_vector
            meld_vector = self.player2_meld_vector

        trick_vector = hand_vector + meld_vector
        device = cfg.DQN_params['device']
        trick_tensor = torch.from_numpy(trick_vector).type(torch.float32).to(device) > 0

        if not is_trick:
            meld_tensor = torch.zeros(vs.MELD_COMBINATIONS_ONE_HOT_VECTOR.__len__(), dtype=torch.uint8)
            hand = self.game.hands[player]
            meld = self.game.melds[player]
            meld_util = MeldUtil(self.game.trump)
            combinations_data = meld_util.combinations

            for i, combo_name in enumerate(combinations_data.keys()):
                meld_tensor[i] = meld_util.is_valid_meld(hand, meld, combo_name)

            # pass is always a valid action, concatenating it at the end of the meld tensor
            meld_pass_tensor = torch.ones(1, dtype=torch.uint8)
            meld_tensor = torch.cat((meld_tensor, meld_pass_tensor))
        else:
            meld_tensor = None

        return trick_tensor, meld_tensor
