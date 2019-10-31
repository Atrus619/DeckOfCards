from util.Constants import Constants as cs


class Trick:
    def __init__(self, players, trump=None):
        self.players = players
        self.trump = trump
        self.cards_in_play = {}
        self.card_scores = {}
        self.initialize_card_scores()

        for player in self.players:
            self.cards_in_play[player] = []

    def initialize_card_scores(self):
        self.card_scores[cs.ACE] = 11
        self.card_scores[cs.TEN] = 10
        self.card_scores[cs.KING] = 4
        self.card_scores[cs.QUEEN] = 3
        self.card_scores[cs.JACK] = 2

    def add_card(self, card, player):
        self.cards_in_play[player].append(card)

    def calculate_trick_score(self, card_1, card_2):
        if card_1.value not in self.card_scores:
            card_1_score = 0
        else:
            card_1_score = self.card_scores[card_1.value]

        if card_2.value not in self.card_scores:
            card_2_score = 0
        else:
            card_2_score = self.card_scores[card_2.value]

        return card_1_score + card_2_score




