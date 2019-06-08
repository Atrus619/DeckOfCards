from util.Constants import Constants as cs


class Trick:
    def __init__(self, players, trump=None):
        self.players = players
        self.trump = trump
        self.cards_in_play = {}

        for player in self.players:
            self.cards_in_play[player] = []

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
            return 1

        if self.trump == card_1.suit:
            if self.trump != card_2.suit:
                return card_1
            elif value_map[card_1.value] >= value_map[card_2.value]:
                return 1
            else:
                return 2

        elif self.trump == card_2.suit:
            return 2

        elif card_1.suit == card_2.suit:
            if value_map[card_1.value] >= value_map[card_2.value]:
                return 1
            else:
                return 2

        else:
            return 1

