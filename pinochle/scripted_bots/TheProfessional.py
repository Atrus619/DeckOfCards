import pinochle.card_util as cu
import util.vector_builder as vb
import numpy as np

class TheProfessional:
    def get_legal_action(self, state, game, player, is_hand):
        """
        Leon, the professional takes care of business. This is our exquisitely hand crafted expert policy.
        Works with trick play only for now.
        :return: Value corresponding to action index of vector corresponding to hand
        """

        largest_card = game.hands[player][0]
        smallest_card = game.hands[player][0]
        for i, card in enumerate(game.hands[player][1:]):
            result_max = cu.compare_cards(game.trump, largest_card, card)
            result_min = cu.compare_cards(game.trump, smallest_card, card)

            if result_max == 1:
                largest_card = card

            if result_min == 0:
                smallest_card = card

        if game.players[game.priority] == player:
            return vb.build_card_vector(largest_card).nonzero()[0][0]
        else:
            return vb.build_card_vector(smallest_card).nonzero()[0][0]

