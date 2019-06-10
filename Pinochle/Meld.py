from classes.Hand import Hand
from util.Util import *
from classes.Card import Card
from util.Constants import Constants as cs

class Meld(Hand):
    def __init__(self, trump):
        Hand.__init__()
        self.trump = trump
        self.combinations = {}
        self.initialize_combinations()

    def show(self):
        pass

    def calculate_score(self, card_list):
        card_dict = list_to_dict(card_list)

        # 28
        # card_dict[28] --> this is the score of the combination
        # if this doesn't exist, its not a valid combination

        # Class A
        # Ace, king, queen, jack, ten trump
        if

        # king, queen of trump

        # king, queen of non trump, same suit

        # dix = nine of trump

        pass

    def validate_move(self):
        pass

    def initialize_combinations(self):
        aceTrump = Card(cs.ACE, self.trump)
        kingTrump = Card(cs.KING, self.trump)
        queenTrump = Card(cs.QUEEN, self.trump)
        jackTrump = Card(cs.JACK, self.trump)
        tenTrump = Card(cs.TEN, self.trump)

        self.combinations["RUN"] = list_to_dict([aceTrump, kingTrump, queenTrump, jackTrump, tenTrump])
        self.combinations["ROYAL_MARRIAGE"] = list_to_dict([kingTrump, queenTrump])
        self.combinations["ROYAL_MARRIAGE"] = list_to_dict([kingTrump, queenTrump])
