import numpy as np


class RandomBot:
    def get_legal_action(self, state, player, game=None, is_hand=None):
        """
        RandomBot just does random actions. Returns a random viable card based on the player's state
        :param state: N/A (placeholders for models that actually use these parameters)
        :param player: Gives information about viable moves
        :param game: N/A
        :param is_hand: N/A
        :return: Value corresponding to action index of vector corresponding to hand
        """
        viable_card_indices = np.where(state.get_player_state(player)[:24] > 0)[0]
        return np.random.choice(viable_card_indices)

    def train_self(self, num_epochs):
        pass

    def copy(self):
        return RandomBot()
