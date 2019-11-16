from classes.Deck import Deck
from pinochle.MeldUtil import MeldUtil
from util.Constants import Constants as cs


class Vectors:
    PINOCHLE_ONE_HOT_VECTOR = Deck('pinochle').return_sorted_deck()

    # the suit makes no difference here since combinations are now suit agnostic
    MELD_COMBINATIONS_ONE_HOT_VECTOR = MeldUtil(cs.HEARTS).combinations.keys()
