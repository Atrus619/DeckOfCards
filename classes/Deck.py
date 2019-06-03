from classes.Suits import Suits
from classes.Value import Value
from classes.Card import Card
from util.Constants import Constants
import numpy as np


class Deck:

    def __init__(self, deck=Constants.STANDARD, num_jokers=0):

        self.deck = deck
        self.cards = []
        assert self.deck in Constants.DECKS, \
            print("Invalid deck configuration requested. Valid configurations include",', '.join([x for x in Constants.DECKS]) + ".")
        eval(Constants.DECKS[self.deck] + '(self)')  # TODO: THIS IS BAD AND SHOULD GET CLEANED UP

    def __len__(self):
        return len(self.cards)

    def init_std_deck(self, num_jokers=0):
        """
        Initializes deck to the standard configuration of 52 cards (13 cards of all 4 suits).
        Can include jokers via optional arg.
        :return: Modifies self.cards in place.
        """
        self.cards = []
        for val in Value.allowed:
            if num_jokers == 0 and val != Constants.JOKER:
                for suit in Suits.allowed:
                    self.cards.append(Card(suit=suit, value=val))
            elif num_jokers > 0 and val == Constants.JOKER:
                for i in range(num_jokers):
                    self.cards.append(Card(suit=None, value=Constants.JOKER))

    def init_pinochle_deck(self):
        """
        Initializes deck to the pinochle configuration.
        See https://en.wikipedia.org/wiki/Pinochle for details.
        :return: Modifies self.cards in place.
        """
        self.cards = []
        allwd_val = np.array((9, 10, Constants.JACK, Constants.QUEEN, Constants.KING, Constants.ACE))
        for val in allwd_val:
            for suit in Suits.allowed:
                self.cards.append(Card(suit=suit, value=val))

    def shuffle(self):
        pass
