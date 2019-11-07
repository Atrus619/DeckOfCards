from util.Constants import Constants as cs


class Trick:
    def __init__(self, players=None, trump=None):
        self.players = players
        self.trump = trump
        self.cards_in_play = {}
        self.card_scores = {}
        self.initialize_card_scores()

        if players is None:
            return

        for player in self.players:
            self.cards_in_play[player] = []

    def initialize_card_scores(self):
        self.card_scores[cs.ACE] = 11
        self.card_scores[cs.TEN] = 10
        self.card_scores[cs.KING] = 4
        self.card_scores[cs.QUEEN] = 3
        self.card_scores[cs.JACK] = 2
        self.card_scores[cs.NINE] = 0

    def add_card(self, card, player):
        self.cards_in_play[player].append(card)

    def calculate_trick_score(self, card_1, card_2):
        return self.card_scores[card_1.value] + self.card_scores[card_2.value]

