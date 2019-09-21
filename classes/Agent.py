from classes.Player import Player
from util.Vectors import Vectors as vs


class Agent(Player):
    def __init__(self, name, model):
        super().__init__(name)
        self.model = model
        self.one_hot_template = vs.PINOCHLE_ONE_HOT_VECTOR

    def get_action(self, state):
        """
        TODO: FILL THIS OUT
        :param state: TODO
        :return: TODO
        """
        return self.model.predict(state)

    def convert_model_output(self, output_index, game, hand=True):
        """
        Converts the model output to a format readable by game
        :param output_index: Integer corresponding to the selected output from the bot. Should map to a specific card's index in a one hot vector.
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
