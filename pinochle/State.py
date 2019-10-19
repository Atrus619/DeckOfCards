import numpy as np
from operator import attrgetter
from util.Vectors import Vectors as vs
from util.util import print_divider
import logging
from config import Config as cfg
import torch

logging.basicConfig(format='%(levelname)s:%(message)s', level=cfg.logging_level)
"""
The strategy is to create a global state with all available information.
Players will only see the portion of the global state that is available to them.
48 cards in the deck (2 each of 24 different cards). First 24 values will correspond to 
a single player's hand, values of 0-2. Next 24 values to the second player's hand.
"""


class State:
    def __init__(self, game):
        self.game = game
        self.one_hot_template = vs.PINOCHLE_ONE_HOT_VECTOR

        player1_hand_vector = self.build_hand_vector(self.game.hands[self.game.players[0]])
        player2_hand_vector = self.build_hand_vector(self.game.hands[self.game.players[1]])
        trump_vector = self.build_trump_vector()

        self.global_state = np.concatenate((player1_hand_vector, player2_hand_vector, trump_vector), axis=0)

    def convert_to_human_readable_format(self, player):
        print_divider()
        self.game.hands[player].show()

        print_divider()
        self.game.melds[player].show()

        current_scores = [self.game.scores[player][-1] for player in self.game.players]
        print_divider()
        logging.debug("Player 1 Score: " + current_scores[0])
        logging.debug("Player 2 Score: " + current_scores[1])

    def get_player_state(self, player):
        """
        Return a subset of the global state corresponding to what the player should be able to see
        :param player:
        :return:
        """
        player_index = self.game.players.index(player)

        player_hand = self.global_state[player_index * len(self.one_hot_template):(player_index + 1) * len(self.one_hot_template)]
        global_info = self.global_state[2 * len(self.one_hot_template):]

        return np.concatenate((player_hand, global_info), axis=0)

    def get_player_state_as_tensor(self, player, device=None):
        arr = self.get_player_state(player)
        device = cfg.DQN_params['device'] if device is None else device
        return torch.from_numpy(arr).type(torch.float32).to(device)

    def get_valid_action_mask(self, player, is_hand):
        """
        Returns a boolean tensor mask that only allows valid action indices
        Assumes first num_action indices are the player's hand
        :param player: Player of interest
        :param is_hand: Whether this is a hand or meld action (boolean, True if hand, False if meld)
        :return: Boolean Tensor
        """
        if is_hand:
            return self.get_player_state_as_tensor(player)[0:cfg.num_actions] > 0
        else:  # Meld
            raise NotImplementedError

    # TODO: move these mf build vectors to their respective classes so we don't have this mess here
    def build_hand_vector(self, hand):
        """
        Produce a one hot vector based on the deck template for a player's hand
        :param hand: List of cards
        :return: NumPy Array
        """

        hand = sorted(hand, key=attrgetter('suit', 'numeric_value'))
        output = np.zeros(len(self.one_hot_template))
        last_index = 0
        for hand_index, hand_card in enumerate(hand):
            if hand_index == 0 or hand_card != hand[hand_index - 1]:
                for template_index, template_card in enumerate(self.one_hot_template):
                    if hand_card == template_card:
                        last_index = template_index
                        output[last_index] = 1
                        break
            else:
                output[last_index] = 2

        return output

    # TODO: move these mf build vectors to their respective classes so we don't have this mess here
    def build_trump_vector(self):
        """
        Produce one hot vector based on the trump card in the current game
        :return:
        """
        suit_template = sorted(set([card.suit for card in self.one_hot_template]))  # TODO: Extremely overkill, might want to hardcode for performance gainz
        output = np.zeros(len(suit_template))

        for index, suit in enumerate(suit_template):
            if suit == self.game.trump:
                output[index] = 1
                return output

    # TODO: move these mf build vectors to their respective classes so we don't have this mess here
    def build_card_vector(self, card):
        """
        Produce a one hot vector based on the deck template for a card
        :param card: card
        :return: NumPy Array
        """

        output = np.zeros(len(self.one_hot_template))
        for template_index, template_card in enumerate(self.one_hot_template):
            if card == template_card:
                last_index = template_index
                output[last_index] = 1
                return output

    # TODO: move these mf build vectors to their respective classes so we don't have this mess here
    def build_meld_cards_vector(self, mt_list):
        """
        Produce a one hot vector based on the deck template for a player's meld cards
        :param mt_list: List of meld tuples
        :return: NumPy Array
        """

        output = np.zeros(len(self.one_hot_template))
        for meld_tuple in mt_list:
            output += self.build_card_vector(meld_tuple[0].card)

        return output
