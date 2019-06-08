from classes.Suits import Suits
from classes.Value import Value
from classes.Card import Card
from util.Constants import Constants as cs
import random
import copy


class Deck:
    def __init__(self, deck=cs.STANDARD, num_jokers=0):
        self.deck = deck.upper()
        self.cards = []

        assert self.deck in cs.DECKS, \
            "Invalid deck configuration requested. Valid configurations include " \
            + ', '.join([x for x in cs.DECKS]) + "."
        if self.deck == cs.STANDARD:
            self.init_std_deck(num_jokers)
        elif self.deck == cs.PINOCHLE:
            assert num_jokers == 0, "Jokers are not part of the typical Pinochle deck."
            self.init_pinochle_deck()

        self.shuffle()

    def __len__(self):
        return len(self.cards)

    def __getitem__(self, item):
        return self.cards[item]

    def init_std_deck(self, num_jokers=0):
        """
        Initializes deck to the standard configuration of 52 cards (13 cards of all 4 suits).
        Can include jokers via optional arg.
        :return: Modifies self.cards in place.
        """
        self.cards = []

        for val in Value.allowed:
            if num_jokers == 0 and val != cs.JOKER:
                for suit in Suits.allowed:
                    self.cards.append(Card(value=val, suit=suit))
            elif num_jokers > 0 and val == cs.JOKER:
                for i in range(num_jokers):
                    self.cards.append(Card(value=cs.JOKER, suit=None))

    def init_pinochle_deck(self):
        """
        Initializes deck to the pinochle configuration.
        See https://en.wikipedia.org/wiki/Pinochle for details.
        :return: Modifies self.cards in place.
        """

        self.cards = []
        allwd_val = {cs.NINE, cs.TEN, cs.JACK, cs.QUEEN, cs.KING, cs.ACE}

        for val in allwd_val:
            for suit in Suits.allowed:
                self.cards.append(Card(value=val, suit=suit))
                self.cards.append(Card(value=val, suit=suit))

    def shuffle(self):
        random.shuffle(self.cards)

    def show_top_cards(self, number):
        """
        Purely for testing, no serious functionality here.
        :param number: Number of cards to show.
        :return: Prints out order of cards from top to bottom, up to n.
        """
        for i, card in enumerate(self.cards):
            if i == number:
                break
            else:
                print(str(i+1) + ":", card)

    def pull_top_cards(self, number_of_cards):
        card_list = []

        for i in range(number_of_cards):
            card_list.append(self.cards[0])
            del(self.cards[0])

        return card_list


