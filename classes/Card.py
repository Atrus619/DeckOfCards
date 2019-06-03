from classes.Suits import Suits
from classes.Value import Value
from util.Constants import Constants


class Card:

    def __init__(self, suit, value):

        if value == Constants.JOKER:
            assert suit is None
        else:
            assert suit in Suits.allowed

        assert value in Value.allowed

        self.value = value
        self.suit = suit

