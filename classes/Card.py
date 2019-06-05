from classes.Suits import Suits
from classes.Value import Value
from util.Constants import Constants as cs


class Card:
    def __init__(self, value, suit):

        self.value = value.upper()
        self.suit = suit.upper()

        if self.value == cs.JOKER:
            assert self.suit is None, "Joker should have suit of None."
        else:
            assert self.suit in Suits.allowed, "Please use a valid suit."

        assert self.value in Value.allowed, "Please use a valid value."

    def __len__(self):
        return 1

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            val_check = self.value == other.value
            suit_check = self.suit == other.suit
            return val_check & suit_check
        else:
            return False

    def __str__(self):
        return self.value + " of " + self.suit
