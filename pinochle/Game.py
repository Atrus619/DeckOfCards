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
import util.state_logger as sl
import logging
from config import Config as cfg
import pinochle.card_util as cu
import time
import pandas as pd
import util.db as db

logging.basicConfig(format='%(levelname)s:%(message)s', level=cfg.logging_level)


# pinochle rules: https://www.pagat.com/marriage/pin2hand.html
class Game:
    def __init__(self, name, players, run_id="42069", current_cycle=None, human_test=False, config=cfg):
        # Setting run_id = None results in no records being saved to database
        self.run_id = run_id
        self.name = name.upper()
        self.players = players  # This is a list
        self.number_of_players = len(self.players)
        self.dealer = players[0]
        self.trump_card = None
        self.trump = None
        self.priority = random.randint(0, 1)
        self.meld_util = None
        self.current_cycle = current_cycle  # To determine the current value of epsilon
        self.human_test = human_test
        self.config = config
        self.exp_df = pd.DataFrame(columns=['agent_id', 'opponent_id', 'run_id', 'vector', 'action', 'next_vector', 'reward'])

        if self.name == cs.PINOCHLE:
            self.deck = Deck("pinochle")
        else:
            self.deck = Deck()

        self.hands = {}
        self.melds = {}
        self.scores = {}
        self.meldedCards = {}
        self.discard_pile = Hand()

        self.player_inter_trick_history = {}  # One entry per player, each entry is a tuple containing (prior_state, row_id entry in initial db update)

        for player in self.players:
            self.hands[player] = Hand()
            self.melds[player] = Meld()
            self.scores[player] = [0]
            self.meldedCards[player] = {}

    def create_state(self, played_card=None):
        return State(self, played_card)

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
            if self.human_test:
                logging.debug("Model hand before action:")
                state.convert_to_human_readable_format(player)

            action = player.get_action(state, self, current_cycle=self.current_cycle, is_hand=True)
            user_input = player.convert_model_output(output_index=action, game=self, is_hand=True)
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
        :param state: Current state of game
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
                action = player.get_action(state, current_cycle=self.current_cycle, is_hand=False)
                user_input = player.convert_model_output(output_index=action, game=self, is_hand=False)

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

    def play_trick(self):
        """
        priority: 0 or 1 for index in player list
        :return: index of winner (priority for next trick)
        """
        print_divider()
        logging.debug(f'Phase 1\tTrick #{12 - len(self.deck)//2}\t{len(self.deck)} card{"s" if len(self.deck) > 1 else ""} remaining in deck')

        trick_start_state = self.create_state()

        trick = Trick(self.players, self.trump)

        # Determine which player goes first based on priority arg
        """ !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # TRICK PLAYER LIST IS NOT ALWAYS THE SAME AS THE GAME PLAYER LIST
        # THEY COULD BE IN DIFFERENT ORDER
        """
        player_order = list(self.players)
        player_1 = player_order.pop(self.priority)
        player_2 = player_order[0]
        trick_player_list = [player_1, player_2]

        # Collect card for trick from each player based on order
        card_1 = self.collect_trick_cards(player_1, trick_start_state)  # Collect card from first player based on priority

        if self.human_test:
            time.sleep(cfg.human_test_pause_length)

        # Recording the first card that was played
        first_move_state = self.create_state(card_1)

        if self.human_test and 'get_Qs' in dir(self.players[0].model):
            print_divider()
            bot_state = trick_start_state if self.players[0] == player_1 else first_move_state
            human_state = trick_start_state if self.players[1] == player_1 else first_move_state
            logging.debug(self.players[0].model.get_Qs(player=self.players[0], player_state=bot_state, opponent=self.players[1], opponent_state=human_state))

        if self.players[0] in self.player_inter_trick_history and self.run_id is not None:  # Don't update on first trick of game
            p1_update_dict = {'player': player_1, 'state_1': self.player_inter_trick_history[player_1][0], 'state_2': trick_start_state, 'row_id': self.player_inter_trick_history[player_1][1]}
            p2_update_dict = {'player': player_2, 'state_1': self.player_inter_trick_history[player_2][0], 'state_2': first_move_state, 'row_id': self.player_inter_trick_history[player_2][1]}
            self.exp_df = sl.update_state(df=self.exp_df, p1=p1_update_dict, p2=p2_update_dict, win_reward=self.config.win_reward)

        card_2 = self.collect_trick_cards(player_2, first_move_state)  # Collect card from second player based on priority

        if self.human_test:
            time.sleep(cfg.human_test_pause_length)

        print_divider()
        logging.debug("LETS GET READY TO RUMBLE!!!!!!!!!!!!!!!!!!!!!!!")
        logging.debug("Card 1: " + str(card_1))
        logging.debug("Card 2: " + str(card_2))

        if self.human_test:
            time.sleep(cfg.human_test_pause_length)

        # Determine winner of trick based on collected cards
        result = cu.compare_cards(self.trump, card_1, card_2)
        print_divider()
        logging.debug("VICTOR : " + str(player_1.name if result == 0 else player_2.name))

        if self.human_test:
            time.sleep(cfg.human_test_pause_length)

        # Separate winner and loser for scoring, melding, and next hand
        winner = trick_player_list.pop(result)
        loser = trick_player_list[0]

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
        mt_list = []
        # no melding in this version
        # while 1:
        #     mt_list, valid = self.collect_meld_cards(winner, meld_state)
        #     if valid:
        #         break
        #     else:
        #         print_divider()
        #         logging.debug("Invalid combination submitted, please try again.")

        # Update scores
        if len(mt_list) == 0:  # No cards melded, so score is 0
            meld_score = 0
        else:
            meld_score = mt_list[0].score  # Score is the same for all MeldTuples in mt_list
        trick_score = trick.calculate_trick_score(card_1, card_2)
        total_score = meld_score + trick_score

        self.discard_pile.add_cards([card_1, card_2])

        # log states and actions, player order = TRICK ORDER
        if self.run_id is not None:
            p1_dict = {'player': player_1, 'state': trick_start_state, 'card': card_1}
            p2_dict = {'player': player_2, 'state': first_move_state, 'card': card_2}
            meld_dict = {'player': winner, 'state': meld_state, 'mt_list': mt_list}
            self.exp_df, self.player_inter_trick_history[player_1], self.player_inter_trick_history[player_2] = sl.log_state(df=self.exp_df, p1=p1_dict, p2=p2_dict, meld=meld_dict, run_id=self.run_id)

        self.scores[winner].append(self.scores[winner][-1] + total_score)
        self.scores[loser].append(self.scores[loser][-1])

        # Update winner's meld
        for mt in mt_list:
            self.melds[winner].add_melded_card(mt)

        # set new priority
        self.priority = self.players.index(winner)

    def play(self):
        while len(self.deck) > 0:
            self.play_trick()

        final_scores = [self.scores[player][-1] for player in self.players]
        winner_index = np.argmax(final_scores)

        if self.run_id is not None:
            # GAME ORDER (because it doesn't matter here)
            end_game_state = self.create_state()
            p1_update_dict = {'player': self.players[0], 'state_1': self.player_inter_trick_history[self.players[0]][0], 'state_2': end_game_state,
                              'row_id': self.player_inter_trick_history[self.players[0]][1]}
            p2_update_dict = {'player': self.players[1], 'state_1': self.player_inter_trick_history[self.players[1]][0], 'state_2': end_game_state,
                              'row_id': self.player_inter_trick_history[self.players[1]][1]}
            self.exp_df = sl.update_state(df=self.exp_df, p1=p1_update_dict, p2=p2_update_dict, winner=self.players[winner_index], win_reward=self.config.win_reward)

        print_divider()
        logging.debug("Winner: " + str(self.players[winner_index]) + "\tScore: " + str(final_scores[winner_index]))
        logging.debug(
            "Loser: " + str(self.players[1 - winner_index]) + "\tScore: " + str(final_scores[1 - winner_index]))
        return winner_index, None if self.run_id is None else self.exp_df
