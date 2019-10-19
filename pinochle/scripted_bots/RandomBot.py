import numpy as np


class RandomBot:
    def __init__(self):
        self.player = None

    def get_legal_action(self, state):
        """
        RandomBot just does random actions. Returns a random viable card based on the player's state
        :param state: N/A
        :return: Value corresponding to action index of vector corresponding to hand
        """
        viable_card_indices = np.where(state.get_player_state(self.player)[:-4] > 0)[0]
        return np.random.choice(viable_card_indices)

    def assign_player(self, player):
        self.player = player

    def train_self(self, num_epochs):
        pass

    def copy(self):
        return RandomBot()
