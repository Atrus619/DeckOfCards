from classes.Player import Player


class Agent(Player):
    def __init__(self, name, model):
        super().__init__(name)
        self.model = model

    def get_action(self, state, msg=None):
        """
        TODO: FILL THIS OUT
        :param state: TODO
        :param msg: Purposefully ignored
        :return: TODO
        """
        return self.model.predict(state)

    def convert_model_output(self, output_index):
        # TODO
        # random bot returns 0 to 23
        # use one_hot_template to determine card
        # find card in hand
        # use card
        pass
