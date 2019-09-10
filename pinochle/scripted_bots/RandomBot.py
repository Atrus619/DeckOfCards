import numpy as np


class RandomBot:
    def __init__(self, length_action_vector):
        self.length_action_vector = length_action_vector
        self.player = None

    def predict(self, state):
        """
        RandomBot just does random actions. Returns a random viable card based on the player's state
        :param state: N/A
        :return: Value corresponding to action index of vector corresponding to hand
        """
        viable_card_indices = np.where(state.get_player_state(self.player)[:-4] > 0)[0]
        return np.random.choice(viable_card_indices)

    def assign_player(self, player):
        self.player = player
