from util.Util import *
from classes.Card import Card
from classes.Suits import Suits
from util.Constants import Constants as cs


class Meld:
    def __init__(self, trump):
        self.trump = trump
        self.combinations = {}
        self.initialize_combinations()

    def calculate_score(self, card_list):
        card_dict = list_to_dict(card_list)

        for combo in self.combinations:
            if type(self.combinations[combo]) is list:
                # this is a special case for marriage
                # marriages are stored in a list of tuples,
                # first value is the dict of the marriage
                # second value is the score of that marriage
                for marriage in self.combinations[combo]:
                    if card_dict == marriage[0]:
                        return marriage[1], marriage[2]
            else:
                if card_dict == self.combinations[combo][0]:
                    return self.combinations[combo][1], self.combinations[combo][2]

        return 0, "NA"

    def validate_meld(self):
        pass

    def initialize_combinations(self):
        # Class A
        ace_trump = Card(cs.ACE, self.trump)
        king_trump = Card(cs.KING, self.trump)
        queen_trump = Card(cs.QUEEN, self.trump)
        jack_trump = Card(cs.JACK, self.trump)
        ten_trump = Card(cs.TEN, self.trump)

        self.combinations["RUN"] = (list_to_dict([ace_trump, king_trump, queen_trump, jack_trump, ten_trump]), 150, "A")
        self.combinations["ROYAL_MARRIAGE"] = (list_to_dict([king_trump, queen_trump]), 40, "A")
        self.combinations["DIX"] = (list_to_dict([Card(cs.NINE, self.trump)]), 10, "A")

        list_of_marriages = []
        remaining_suits = [suit for suit in Suits.allowed if suit != self.trump]
        self.combinations["MARRIAGE"] = list_of_marriages

        for suit in remaining_suits:
            list_of_marriages.append((list_to_dict([Card(cs.KING, suit), Card(cs.QUEEN, suit)]), 20, "A"))

        # Class B
        for value in [(cs.ACE, 100), (cs.KING, 80), (cs.QUEEN, 60), (cs.JACK, 40, "B")]:
            list_around = []

            for suit in Suits.allowed:
                list_around.append(Card(value[0], suit))

            self.combinations[value[0] + "_AROUND"] = (list_to_dict(list_around), value[1], "B")

        # Class C
        pinochle_list = [Card(cs.QUEEN, cs.SPADES), Card(cs.JACK, cs.DIAMONDS)]
        self.combinations["PINOCHLE"] = (list_to_dict(pinochle_list), 40, "C")

        double_pinochle_list = pinochle_list + pinochle_list
        self.combinations["DOUBLE_PINOCHLE"] = (list_to_dict(double_pinochle_list), 300, "C")

