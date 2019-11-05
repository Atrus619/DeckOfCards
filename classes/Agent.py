from classes.Player import Player
from util.Vectors import Vectors as vs
import random
from pinochle.scripted_bots.RandomBot import RandomBot


class Agent(Player):
    def __init__(self, name, model, epsilon):
        super().__init__(name)
        self.model = model
        self.one_hot_template = vs.PINOCHLE_ONE_HOT_VECTOR
        self.epsilon = epsilon
        self.random_bot = RandomBot()

    def get_action(self, state, game, is_hand, current_cycle):
        """
        Retrieves the index of a legal action from the model. With probability epsilon, will take a random action.
        :param state: Current state of the game
        :param game: game details
        :param is_hand: Boolean corresponding to whether this is a hand action or meld action
        :param current_cycle: Current cycle in training process, used to determine value of epsilon
        :return: Index of action corresponding to state.one_hot_vector of cards
        """
        epsilon = self.epsilon.get_epsilon(current_cycle=current_cycle)
        if random.random() > epsilon:
            return self.model.get_legal_action(state=state, player=self, game=game, is_hand=is_hand) if is_hand else None  # TODO: Implement meld later
        else:
            return self.random_bot.get_legal_action(state=state, player=self)

    def convert_model_output(self, output_index, game, is_hand=True):
        """
        Converts the model output to a format readable by game
        :param output_index: Integer corresponding to the selected output from the bot. Should map to a specific card's index in a one hot vector.
        :param game: current game object to access properties
        :param is_hand: If false, then meld implied
        :return: Game expected input
        """
        if not is_hand:  # TODO: Remove this later, it is a simplification to skip melding
            return 'Y'

        selected_card = self.one_hot_template[output_index]

        leading_char = 'H' if is_hand else 'M'

        for card in game.hands[self]:
            if selected_card == card:
                return leading_char + str(game.hands[self].cards.index(card))

    def set_model(self, model):
        self.model = model
