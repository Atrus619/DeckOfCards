from util.util import *
from classes.Card import Card
from util.Constants import Constants as cs
from collections import OrderedDict
from copy import deepcopy


class MeldUtil:
    def __init__(self, trump):
        self.trump = trump
        self.combinations = OrderedDict()
        self.initialize_combinations()

    def calculate_score(self, card_list):
        """
        Calculates the meld score
        :param card_list: INSIGNIFICANT.
        :return: (score, meld class, combination name)
        """
        card_dict = list_to_dict(card_list)

        for combo in self.combinations:
            if card_dict == self.combinations[combo][0]:
                return self.combinations[combo][1], self.combinations[combo][2], combo

        return 0, "NA", "NA"

    def initialize_combinations(self):
        # Class A
        ace_trump = Card(cs.ACE, self.trump)
        king_trump = Card(cs.KING, self.trump)
        queen_trump = Card(cs.QUEEN, self.trump)
        jack_trump = Card(cs.JACK, self.trump)
        ten_trump = Card(cs.TEN, self.trump)

        self.combinations["RUN"] = (list_to_dict([ace_trump, king_trump, queen_trump, jack_trump, ten_trump]), 150, "A")
        self.combinations["DIX"] = (list_to_dict([Card(cs.NINE, self.trump)]), 10, "A")

        for suit in cs.SUITS:
            self.combinations["MARRIAGE_" + suit] = \
                (list_to_dict([Card(cs.KING, suit), Card(cs.QUEEN, suit)]), 40 if suit == self.trump else 20, "A")

        # Class B
        for value in [(cs.ACE, 100), (cs.KING, 80), (cs.QUEEN, 60), (cs.JACK, 40, "B")]:
            list_around = []

            for suit in cs.SUITS:
                list_around.append(Card(value[0], suit))

            self.combinations[value[0] + "_AROUND"] = (list_to_dict(list_around), value[1], "B")

        # Class C
        pinochle_list = [Card(cs.QUEEN, cs.SPADES), Card(cs.JACK, cs.DIAMONDS)]
        self.combinations["PINOCHLE"] = (list_to_dict(pinochle_list), 40, "C")

        double_pinochle_list = pinochle_list + pinochle_list
        self.combinations["DOUBLE_PINOCHLE"] = (list_to_dict(double_pinochle_list), 300, "C")

    def is_valid_meld(self, hand, meld, combo_name):
        combo_cards_dict = self.combinations[combo_name][0]
        combo_score = self.combinations[combo_name][1]
        combo_class = self.combinations[combo_name][2]
        hand_card_present = False
        hand_copy = deepcopy(hand)
        meld_copy = deepcopy(meld)

        for card, count in combo_cards_dict.items():
            # only loop twice for double pinochle
            for i in range(count):
                # check card presence in hand because one card has to come from hand
                if card in hand_copy:
                    hand_card_present = True
                    hand_copy.pull_card(card)
                    continue

                # if not present in hand, check meld
                meld_tuple = meld_copy.get_mt_if_card_present(card)
                if meld_tuple:
                    # removing this card from further consideration
                    meld_copy.pull_melded_card(meld_tuple)
                    if meld_tuple.meld_class != combo_class or meld_tuple.score < combo_score:
                        # meld is valid if class is different or score is higher
                        continue
                    else:
                        # if first pulled card from meld does not satisfy meld requirements,
                        # check if a second, same card exists and satisfies requirements
                        meld_tuple = meld_copy.get_mt_if_card_present(card)
                        if meld_tuple:
                            meld_copy.pull_melded_card(meld_tuple)
                            if meld_tuple.meld_class != combo_class or meld_tuple.score < combo_score:
                                continue

                return False

        return hand_card_present

    def generate_combo(self, hand, meld, combo_name):
        """
        :param hand:
        :param meld:
        :param combo_name:
        :return:
        Modifies hand and meld and returns the list of cards that will create the new combo
        """
        combo_cards_dict = self.combinations[combo_name][0]
        combo_score = self.combinations[combo_name][1]
        combo_class = self.combinations[combo_name][2]
        meld_copy = deepcopy(meld)
        meld_cards = []
        hand_cards = []

        for card, count in combo_cards_dict.items():
            # only loop twice for double pinochle
            for i in range(count):
                # check meld
                meld_tuple = meld_copy.get_mt_if_card_present(card)
                if meld_tuple:
                    # removing this card from further consideration
                    meld_copy.pull_melded_card(meld_tuple)
                    if meld_tuple.meld_class != combo_class or meld_tuple.score < combo_score:
                        # meld is valid if class is different or score is higher
                        meld_cards.append(meld_tuple.card)
                        continue
                    else:
                        # if first pulled card from meld does not satisfy meld requirements,
                        # check if a second, same card exists and satisfies requirements
                        meld_tuple = meld_copy.get_mt_if_card_present(card)
                        if meld_tuple:
                            meld_copy.pull_melded_card(meld_tuple)
                            if meld_tuple.meld_class != combo_class or meld_tuple.score < combo_score:
                                meld_cards.append(meld_tuple.card)
                                continue

                hand_cards.append(card)

        # check if there are no hand cards present so far
        # if none present, take a card from meld
        if hand_cards.__len__():
            for card in meld_cards:
                if card in hand:
                    hand_cards.append(card)
                    meld_cards.remove(card)
                    break

        # reset meld copy, remove final list of meld cards
        meld_copy = deepcopy(meld)
        for card in meld_cards:
            card_id = id(card)

            for mt in meld_copy.melded_cards:
                if card_id == id(mt.card):
                    meld.pull_melded_card(mt)

        hand.drop_cards(hand_cards)

        collected_cards = hand_cards + meld_cards
        score, meld_class, combo_name = self.calculate_score(collected_cards)

        return score, meld_class, combo_name, collected_cards

