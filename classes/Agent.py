from classes.Player import Player
from util.Vectors import Vectors as vs
import random
from pinochle.scripted_bots.RandomBot import RandomBot
from pinochle.MeldUtil import MeldUtil


class Agent(Player):
    def __init__(self, name, model, epsilon):
        super().__init__(name)
        self.model = model
        self.one_hot_template = vs.PINOCHLE_ONE_HOT_VECTOR
        self.epsilon = epsilon
        self.random_bot = RandomBot()

    def get_action(self, state, game, is_trick, current_cycle):
        """
        Retrieves the index of a legal action from the model. With probability epsilon, will take a random action.
        :param state: Current state of the game
        :param game: game details
        :param is_trick: Boolean corresponding to whether this is a trick action or meld action
        :param current_cycle: Current cycle in training process, used to determine value of epsilon
        :return: Index of action corresponding to state.one_hot_vector of cards
        """
        epsilon = self.epsilon.get_epsilon(current_cycle=current_cycle)
        if random.random() > epsilon:
            return self.model.get_legal_action(state=state, player=self, game=game, is_trick=is_trick) if is_trick else None
        else:
            return self.random_bot.get_legal_action(state=state, player=self, game=game, is_trick=is_trick)

    def convert_model_output(self, trick_index, meld_index, game, is_trick=True):
        """
        Converts the model output to a format readable by game
        :param output_index: Integer corresponding to the selected output from the bot. Should map to a specific card's index in a one hot vector.
        :param game: current game object to access properties
        :param is_trick: If false, then meld implied
        :return: Game expected input
        """
        trick_output = None
        selected_card = self.one_hot_template[trick_index]

        for card in game.hands[self]:
            if selected_card == card:
                trick_output = 'H' + str(game.hands[self].cards.index(card))
                break

        if is_trick:
            return trick_output, None

        combo_name = vs.MELD_COMBINATIONS_ONE_HOT_VECTOR[meld_index]
        return MeldUtil.generate_combo(game.hands[self], game.melds[self], combo_name)

    def set_model(self, model):
        self.model = model
