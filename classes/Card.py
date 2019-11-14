from classes.Value import Value
from util.Constants import Constants as cs


class Card:
    def __init__(self, value, suit):

        self.value = value.upper()
        self.suit = suit.upper()
        self.numeric_value = cs.STANDARD_CARDS_VALUE[self.value]

        if self.value == cs.JOKER:
            assert self.suit is None, "Joker should have suit of None."
        else:
            assert self.suit in cs.suits, "Please use a valid suit."

        assert self.value in Value.allowed, "Please use a valid value."

    def __len__(self):
        return 1

    def __eq__(self, other):
        return (self.value, self.suit) == (other.value, other.suit)

    def __str__(self):
        return self.value + " of " + self.suit

    def __hash__(self):
        return hash((self.value, self.suit))
