from classes.Suits import Suits
from classes.Value import Value


class Card:
    suit = None

    value = None

    def __init__(self, suit, value):

        assert suit in Suits.allowed
        assert value in Value.allowed

        self.suit = suit
        self.value = value
