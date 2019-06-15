from classes.Card import Card
from classes.Deck import Deck
from classes.Hand import Hand
from pinochle.Meld import Meld
from pinochle.Trick import Trick
from util.Constants import Constants as cs
from copy import deepcopy


# pinochle rules: https://www.pagat.com/marriage/pin2hand.html
class Game:
    def __init__(self, name, players):
        self.name = name.upper()
        self.players = players
        self.number_of_players = len(self.players)
        self.dealer = players[0]
        self.trump_card = None
        self.trump = None
        self.meld = None

        if self.name == cs.PINOCHLE:
            self.deck = Deck("pinochle")
        else:
            self.deck = Deck()

        self.hands = {}
        self.melds = {}
        self.scores = {}

        for player in self.players:
            self.hands[player] = Hand()
            self.melds[player] = Hand()
            self.scores[player] = [0]

    def deal(self):
        for i in range(12):
            for player in self.players:
                self.hands[player].add_cards(self.deck.pull_top_cards(1))

        self.trump_card = self.deck.pull_top_cards(1)[0]
        self.trump = self.trump_card.suit
        self.meld = Meld(self.trump)

    # Expected card input: VALUE,SUIT. Example: HAND,ACE,CLUBS
    def collect_trick_cards(self, player):
        self.hands[player].show("hand")
        self.melds[player].show("meld")

        user_input = input(player.name + " select card for trick:")
        source, value, suit = user_input.split(',')
        card_input = Card(value, suit)

        if source == "HAND":
            card = self.hands[player].pull_card(card_input)
        elif source == "MELD":
            card = self.melds[player].pull_card(card_input)

        return card

    def collect_meld_cards(self, player, limit=12):
        first_hand_card = True
        valid = True
        original_hand_cards = deepcopy(self.hands[player])
        original_meld_cards = deepcopy(self.melds[player])
        collected_cards = []

        while len(collected_cards) < limit:
            self.hands[player].show("hand")
            self.melds[player].show("meld")

            if first_hand_card:
                print("For meld please select first card from hand.")

            user_input = input(player.name + " select card, type 'Y' to exit:")

            if user_input == 'Y':
                break

            source, value, suit = user_input.split(',')
            if first_hand_card:
                if source != "HAND":
                    print("In case of meld, please select first card from hand.")
                    continue

                first_hand_card = False

            card_input = Card(value, suit)

            if source == "HAND":
                card = self.hands[player].pull_card(card_input)
            elif source == "MELD":
                card = self.melds[player].pull_card(card_input)

            collected_cards.append(card)

        if len(collected_cards) > 0:
            score = self.meld.calculate_score(collected_cards)

            if score == 0:
                self.hands[player] = original_hand_cards
                self.melds[player] = original_meld_cards
                valid = False

        return score, valid

    def play_trick(self, priority):
        """
        :param priority: 0 or 1 for index in player list
        :return:
        """
        trick = Trick(self.players, self.trump)
        player_order = list(self.players)
        player_1 = player_order.pop(priority)
        player_2 = player_order[0]

        card_1 = self.collect_trick_cards(player_1, 1)
        card_2 = self.collect_trick_cards(player_2, 1)

        # TODO: make all players see all cards played
        print("LETS GET READY TO RUMBLE!!!!!!!!!!!!!!!!!!!!!!!")
        print("Card 1: " + str(card_1))
        print("Card 2: " + str(card_2))

        result = trick.compare_cards(card_1, card_2)

        print("VICTOR : " + str(result))

        if result == 0:
            print(player_1.name + " select cards for meld:")

            while 1:
                meld, valid = self.collect_cards(player_1)
                if valid:
                    print("Invalid combination submitted, please try again.")
                    break


            # TODO: figure out how to add calculated meld score back in, how to have history of it
            self.melds[player_1].calculate_score(meld)

            self.scores[player_1] += result
        else:
            self.scores[player_2] += result

        pass

