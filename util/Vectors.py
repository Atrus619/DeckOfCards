from classes.Deck import Deck
from pinochle.MeldUtil import MeldUtil
from util.Constants import Constants as cs


class Vectors:
    PINOCHLE_ONE_HOT_VECTOR = Deck('pinochle').return_sorted_deck()

    MELD_COMBINATIONS_ONE_HOT_VECTOR = {}
    for suit in cs.SUITS:
        MELD_COMBINATIONS_ONE_HOT_VECTOR[suit] = MeldUtil(suit).combinations.keys()
