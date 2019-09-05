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
