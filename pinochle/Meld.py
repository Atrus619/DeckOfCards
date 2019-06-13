from classes.Hand import Hand
from util.Util import *
from classes.Card import Card
from classes.Suits import Suits
from util.Constants import Constants as cs


class Meld(Hand):
    def __init__(self, trump):
        super(Meld, self).__init__()
        self.trump = trump
        self.combinations = {}
        self.initialize_combinations()

    def show(self):
        pass

    def calculate_score(self, card_list):
        # take all values from dict
        # in case of marriage, take out the 3 dicts, add to the first list
        # card_dict = list_to_dict(card_list)

        for combo in self.combinations:
            if type(self.combinations[combo][0]) is list:
                if card_list in self.combinations[combo][0]:
                    return self.combinations[combo][1]
            else:
                if card_list == self.combinations[combo][0]:
                    return self.combinations[combo][1]

    def validate_move(self):
        pass

    def initialize_combinations(self):
        # Class A
        ace_trump = Card(cs.ACE, self.trump)
        king_trump = Card(cs.KING, self.trump)
        queen_trump = Card(cs.QUEEN, self.trump)
        jack_trump = Card(cs.JACK, self.trump)
        ten_trump = Card(cs.TEN, self.trump)

        self.combinations["RUN"] = (list_to_dict([ace_trump, king_trump, queen_trump, jack_trump, ten_trump]), 150)
        self.combinations["ROYAL_MARRIAGE"] = (list_to_dict([king_trump, queen_trump]), 40)
        self.combinations["DIX"] = (list_to_dict([Card(cs.NINE, self.trump)]), 10)

        list_of_marriages = []
        remaining_suits = [suit for suit in Suits.allowed if suit != self.trump]
        self.combinations["MARRIAGE"] = list_of_marriages

        for suit in remaining_suits:
            list_of_marriages.append((list_to_dict([Card(cs.KING, suit), Card(cs.QUEEN, suit)]), 20))

        # Class B
        for value in [(cs.ACE, 100), (cs.KING, 80), (cs.QUEEN, 60), (cs.JACK, 40)]:
            list_around = []

            for suit in Suits.allowed:
                list_around.append(Card(value[0], suit))

            self.combinations[value[0] + "_AROUND"] = (list_to_dict(list_around), value[1])

        # Class C
        pinochle_list = [hash(Card(cs.QUEEN, cs.SPADES)), hash(Card(cs.JACK, cs.DIAMONDS))]
        self.combinations["PINOCHLE"] = (list_to_dict(pinochle_list), 40)

        double_pinochle_list = pinochle_list + pinochle_list
        self.combinations["DOUBLE_PINOCHLE"] = (list_to_dict(double_pinochle_list), 300)

