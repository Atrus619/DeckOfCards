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

    # def evaluate(self, cards):
    #     for player in self.players:

    def compare_cards(self, card_1, card_2):
        """
        :param card_1:
        :param card_2: 
        :return: the number that represents the card that won
        """
        value_map = cs.STANDARD_CARDS_VALUE

        if card_1 == card_2:
            return 0

        if self.trump == card_1.suit:
            if self.trump != card_2.suit:
                return 0
            elif value_map[card_1.value] >= value_map[card_2.value]:
                return 0
            else:
                return 1

        elif self.trump == card_2.suit:
            return 1

        elif card_1.suit == card_2.suit:
            if value_map[card_1.value] >= value_map[card_2.value]:
                return 0
            else:
                return 1

        else:
            return 0

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




