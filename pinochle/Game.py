from pinochle.State import State
from classes.Deck import Deck
from classes.Hand import Hand
from pinochle.MeldUtil import MeldUtil
from pinochle.Meld import Meld
from pinochle.Trick import Trick
from pinochle.MeldTuple import MeldTuple
from util.Constants import Constants as cs
from util.util import print_divider
from copy import deepcopy
import random
import numpy as np
from util.state_logger import log_state
import logging
from config import Config as cfg

logging.basicConfig(format='%(levelname)s:%(message)s', level=cfg.logging_level)


# pinochle rules: https://www.pagat.com/marriage/pin2hand.html
class Game:
    def __init__(self, name, players, run_id="42069"):
        self.run_id = run_id
        self.name = name.upper()
        self.players = players  # This is a list
        self.number_of_players = len(self.players)
        self.dealer = players[0]
        self.trump_card = None
        self.trump = None
        self.meld_util = None

        if self.name == cs.PINOCHLE:
            self.deck = Deck("pinochle")
        else:
            self.deck = Deck()

        self.hands = {}
        self.melds = {}
        self.scores = {}
        self.meldedCards = {}

        for player in self.players:
            self.hands[player] = Hand()
            self.melds[player] = Meld()
            self.scores[player] = [0]
            self.meldedCards[player] = {}

    def create_state(self):
        return State(self)

    def deal(self):
        for i in range(12):
            for player in self.players:
                self.hands[player].add_cards(self.deck.pull_top_cards(1))

        self.trump_card = self.deck.pull_top_cards(1)[0]
        self.trump = self.trump_card.suit
        self.meld_util = MeldUtil(self.trump)

    # Expected card input: VALUE,SUIT. Example: Hindex
    # H = hand, M = meld
    def collect_trick_cards(self, player, state):
        if type(player).__name__ == 'Human':
            user_input = player.get_action(state, msg=player.name + " select card for trick:")
        else:  # Bot
            action = player.get_action(state)
            user_input = player.convert_model_output(output_index=action, game=self, hand=True)
        source = user_input[0]
        index = int(user_input[1:])

        if source == "H":
            card_input = self.hands[player].cards[index]
            card = self.hands[player].pull_card(card_input)
        elif source == "M":
            mt = self.melds[player].pull_melded_card(self.melds[player].melded_cards[index])
            card = mt.card

        print_divider()
        logging.debug("Player " + player.name + " plays: " + str(card))  # TODO: Fix this later (possible NULL)
        return card

    def collect_meld_cards(self, player, state, limit=12):
        """
        Collecting cards for meld scoring from player who won trick
        :param player: Player we are collecting from
        :param limit: Maximum number of cards that can be collected
        :return: list of MeldTuples and whether the interaction was valid (boolean)
        """
        first_hand_card = True
        valid = True
        original_hand_cards = deepcopy(self.hands[player])
        original_meld_cards = deepcopy(self.melds[player])
        collected_hand_cards = []
        collected_meld_cards = []
        score = 0
        meld_class = None
        combo_name = None

        while len(collected_hand_cards) + len(collected_meld_cards) < limit:
            print_divider()
            self.hands[player].show()
            print_divider()
            self.melds[player].show()

            if first_hand_card:
                print_divider()
                logging.debug("For meld please select first card from hand.")

            if type(player).__name__ == 'Human':
                user_input = player.get_action(state, msg=player.name + " select card, type 'Y' to exit:")
            else:  # Bot
                action = player.get_action(state)
                user_input = player.convert_model_output(output_index=action, game=self, hand=False)

            if user_input == 'Y':
                break

            source = user_input[0]
            index = int(user_input[1:])

            if first_hand_card:
                if source != "H":
                    print_divider()
                    logging.debug("In case of meld, please select first card from hand.")
                    continue

                first_hand_card = False

            if source == "H":
                card_input = self.hands[player].cards[index]
                card = self.hands[player].pull_card(card_input)
                collected_hand_cards.append(card)
            elif source == "M":
                mt = self.melds[player].pull_melded_card(self.melds[player].melded_cards[index])
                collected_meld_cards.append(mt)

        # Combine collected hand and meld card lists for score calculation
        collected_cards = collected_hand_cards + [mt.card for mt in collected_meld_cards]

        if len(collected_cards) > 0:
            score, meld_class, combo_name = self.meld_util.calculate_score(collected_cards)

            if score == 0:
                valid = False
            else:
                for mt in collected_meld_cards:
                    original_meld_class = mt.meld_class
                    if original_meld_class == meld_class:
                        original_meld_score = mt.score
                        if original_meld_score <= score:
                            valid = False
                            break
            if not valid:
                self.hands[player] = original_hand_cards
                self.melds[player] = original_meld_cards

        return [MeldTuple(card, combo_name, meld_class, score) for card in collected_cards], valid

    def play_trick(self, priority):
        """
        :param priority: 0 or 1 for index in player list
        :return: index of winner (priority for next trick)
        """
        trick_state = self.create_state()

        trick = Trick(self.players, self.trump)

        # Determine which player goes first based on priority arg
        player_order = list(self.players)
        player_1 = player_order.pop(priority)
        player_2 = player_order[0]

        # Collect card for trick from each player based on order
        card_1 = self.collect_trick_cards(player_1, trick_state)
        card_2 = self.collect_trick_cards(player_2, trick_state)

        # TODO: make all players see all cards played
        print_divider()
        logging.debug("LETS GET READY TO RUMBLE!!!!!!!!!!!!!!!!!!!!!!!")
        logging.debug("Card 1: " + str(card_1))
        logging.debug("Card 2: " + str(card_2))

        # Determine winner of trick based on collected cards
        result = trick.compare_cards(card_1, card_2)
        print_divider()
        logging.debug("VICTOR : " + str(player_1.name if result == 0 else player_2.name))

        # Separate winner and loser for scoring, melding, and next hand
        copy_of_players = list(self.players)
        winner = copy_of_players.pop(result)
        loser = copy_of_players[0]

        # Winner draws a card from the stock, followed by the loser drawing a card from the stock
        # TODO: Come back here and allow winner to choose when down to last 2 cards (optional af)
        self.hands[winner].add_cards(self.deck.pull_top_cards(1))
        if len(self.deck) == 0:
            self.hands[loser].add_cards(self.trump_card)
        else:
            self.hands[loser].add_cards(self.deck.pull_top_cards(1))

        # Winner can now meld if they so choose
        print_divider()
        logging.debug(winner.name + " select cards for meld:")

        # Verify that meld is valid. If meld is invalid, force the user to retry.
        meld_state = self.create_state()

        while 1:
            mt_list, valid = self.collect_meld_cards(winner, meld_state)
            if valid:
                break
            else:
                print_divider()
                logging.debug("Invalid combination submitted, please try again.")

        # Update scores
        if len(mt_list) == 0:  # No cards melded, so score is 0
            meld_score = 0
        else:
            meld_score = mt_list[0].score  # Score is the same for all MeldTuples in mt_list
        trick_score = trick.calculate_trick_score(card_1, card_2)
        total_score = meld_score + trick_score

        # log states and actions
        log_state(trick_state, meld_state, card_1, card_2,
                  mt_list, trick_score, meld_score, winner, self.players[0], self.players[1], self.run_id)

        self.scores[winner].append(self.scores[winner][-1] + total_score)
        self.scores[loser].append(self.scores[loser][-1])

        # Update winner's meld
        for mt in mt_list:
            self.melds[winner].add_melded_card(mt)

        return result

    def play(self):
        priority = random.randint(0, 1)
        while len(self.deck) > 0:
            priority = self.play_trick(priority)
        final_scores = [self.scores[player][-1] for player in self.players]
        winner_index = np.argmax(final_scores)
        print_divider()
        logging.debug("Winner: " + str(self.players[winner_index]) + "\tScore: " + str(final_scores[winner_index]))
        logging.debug("Loser: " + str(self.players[1-winner_index]) + "\tScore: " + str(final_scores[1-winner_index]))
        return winner_index
