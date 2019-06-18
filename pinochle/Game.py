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

    # Expected card input: VALUE,SUIT. Example: Hindex + 1
    # H = hand, M = meld
    def collect_trick_cards(self, player):
        self.hands[player].show("hand")
        self.melds[player].show("meld")

        user_input = input(player.name + " select card for trick:")
        source = user_input[0]
        index = int(user_input[1:]) - 1

        if source == "H":
            card_input = self.hands[player].cards[index]
            card = self.hands[player].pull_card(card_input)
        elif source == "M":
            card_input = self.melds[player].cards[index]
            card = self.melds[player].pull_card(card_input)

        print("returning card: " + card.value + " " + card.suit)
        return card

    def collect_meld_cards(self, player, limit=12):
        first_hand_card = True
        valid = True
        original_hand_cards = deepcopy(self.hands[player])
        original_meld_cards = deepcopy(self.melds[player])
        collected_cards = []
        score = 0

        while len(collected_cards) < limit:
            self.hands[player].show("hand")
            self.melds[player].show("meld")

            if first_hand_card:
                print("For meld please select first card from hand.")

            user_input = input(player.name + " select card, type 'Y' to exit:")

            if user_input == 'Y':
                break

            source = user_input[0]
            index = int(user_input[1:]) - 1

            if first_hand_card:
                if source != "H":
                    print("In case of meld, please select first card from hand.")
                    continue

                first_hand_card = False

            if source == "H":
                card_input = self.hands[player].cards[index]
                card = self.hands[player].pull_card(card_input)
            elif source == "M":
                card_input = self.melds[player].cards[index]
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

        card_1 = self.collect_trick_cards(player_1)
        card_2 = self.collect_trick_cards(player_2)

        # TODO: make all players see all cards played
        print("LETS GET READY TO RUMBLE!!!!!!!!!!!!!!!!!!!!!!!")
        print("Card 1: " + str(card_1))
        print("Card 2: " + str(card_2))

        result = trick.compare_cards(card_1, card_2)

        print("VICTOR : " + str(result))

        copy_of_players = list(self.players)
        winner = copy_of_players.pop(result)
        loser = copy_of_players[0]

        print(winner.name + " select cards for meld:")

        while 1:
            meld_score, valid = self.collect_meld_cards(winner)
            if valid:
                break
            else:
                print("Invalid combination submitted, please try again.")

        trick_score = trick.calculate_trick_score(card_1, card_2)
        total_score = meld_score + trick_score

        self.scores[winner].append(self.scores[winner][-1] + total_score)
        self.scores[loser].append(self.scores[loser][-1])


