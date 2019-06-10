from util.Constants import Constants as cs
from classes.Deck import Deck
from classes.Hand import Hand
from Pinochle.Meld import Meld


# Pinochle rules: https://www.pagat.com/marriage/pin2hand.html
class Game:
    def __init__(self, name, players):
        self.name = name.upper()
        self.players = players
        self.number_of_players = len(self.players)
        self.dealer = players[0]
        self.trump_card = None
        self.trump = None

        if self.name == cs.PINOCHLE:
            self.deck = Deck("pinochle")
        else:
            self.deck = Deck()

        self.hands = {}
        self.melds = {}
        self.scores = {}

        for player in self.players:
            self.hands[player] = Hand()
            self.melds[player] = Meld()
            self.scores[player] = [0]

    def deal(self):
        for i in range(12):
            for player in self.players:
                self.hands[player].add_cards(self.deck.pull_top_cards(1))

        self.trump_card = self.deck.pull_top_cards(1)[0]
        self.trump = self.trump_card.suit

    def next_round(self):
        pass

