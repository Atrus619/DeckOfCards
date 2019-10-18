from classes.Player import Player
from util.Vectors import Vectors as vs
import random
from pinochle.scripted_bots.RandomBot import RandomBot


class Agent(Player):
    def __init__(self, name, model, epsilon_func):
        super().__init__(name)
        self.model = model
        self.one_hot_template = vs.PINOCHLE_ONE_HOT_VECTOR
        self.epsilon_func = epsilon_func
        self.random_bot = RandomBot()
        self.random_bot.assign_player(self)

    def get_action(self, state, current_cycle):
        """
        TODO: FILL THIS OUT
        :param state: TODO
        :return: TODO
        """
        epsilon = self.epsilon_func(current_cycle=current_cycle)
        if random.random() > epsilon:
            return self.model.policy_net(state.get_player_state_as_tensor(player=self)).argmax()
        else:
            return self.random_bot.predict(state)

    def convert_model_output(self, output_index, game, hand=True):
        """
        Converts the model output to a format readable by game
        :param output_index: Integer corresponding to the selected output from the bot. Should map to a specific card's index in a one hot vector.
        :param game: current game object to access properties
        :param hand: If false, then meld implied
        :return: Game expected input
        """
        selected_card = self.one_hot_template[output_index]

        if not hand:  # TODO: Remove this later, it is a simplification to skip melding
            return 'Y'

        leading_char = 'H' if hand else 'M'

        for card in game.hands[self]:
            if selected_card == card:
                return leading_char + str(game.hands[self].cards.index(card))
