import pinochle.card_util as cu
import util.vector_builder as vb
from util.Vectors import Vectors as vs
from pinochle.Trick import Trick


class ExpertPolicy:
    def get_legal_action(self, state, player, game, is_hand):
        """
        Leon, the professional takes care of business. This is our exquisitely hand crafted expert policy.
        Works with trick play only for now.
        :return: Value corresponding to action index of vector corresponding to hand
        """
        played_card = None
        viable_cards = []

        # determine what card was already played
        if state.played_card_vector.sum() > 0:
            played_card = vs.PINOCHLE_ONE_HOT_VECTOR[state.played_card_vector.argmax()]

        largest_card = game.hands[player][0]
        smallest_card = game.hands[player][0]
        for i, card in enumerate(game.hands[player][1:]):
            result_max = cu.compare_cards(game.trump, largest_card, card)
            result_min = cu.compare_cards(game.trump, smallest_card, card)

            if result_max == 1:
                largest_card = card

            if result_min == 0:
                smallest_card = card

            # build list of all cards that will beat an already played card
            if played_card is not None:
                result_first_card = cu.compare_cards(game.trump, card, played_card)
                if result_first_card == 0:
                    viable_cards.append(card)

        # determine highest point value card that can be played to win the trick when going second
        # preferably non trump card
        if len(viable_cards) > 0:
            point_values = Trick()
            optimal_card = viable_cards[0]
            for i, current_card in enumerate(viable_cards):
                optimal_points = point_values.card_scores[optimal_card.value]
                current_points = point_values.card_scores[current_card.value]

                if current_points == optimal_points:
                    if optimal_card.suit == game.trump and current_card.suit != game.trump:
                        optimal_card = current_card
                elif current_points > optimal_points:
                    optimal_card = current_card

        if game.players[game.priority] == player:
            return vb.build_card_vector(largest_card).nonzero()[0][0]
        else:
            if len(viable_cards) == 0:
                return vb.build_card_vector(smallest_card).nonzero()[0][0]
            else:
                return vb.build_card_vector(optimal_card).nonzero()[0][0]

