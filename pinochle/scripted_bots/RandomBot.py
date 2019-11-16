import numpy as np


class RandomBot:
    def get_legal_action(self, state, player, game, is_trick):
        """
        RandomBot just does random actions. Returns a random viable card based on the player's state
        :param state: N/A (placeholders for models that actually use these parameters)
        :param player: Gives information about viable moves
        :param game: N/A
        :param is_trick: N/A
        :return: Value corresponding to action index of vector corresponding to hand
        """
        viable_trick_one_hot_vectors, viable_meld_one_hot_vectors = state.get_valid_action_mask(player, is_trick)
        trick_index = np.random.choice(np.where(viable_trick_one_hot_vectors > 0)[0])


        if is_trick:
            return trick_index, None

        viable_meld_indices = np.where(viable_trick_one_hot_vectors > 0)[0]

        if len(viable_meld_indices) > 0:
            return trick_index, np.random.choice(viable_meld_indices)

        return

    def train_self(self, num_epochs):
        pass

    def copy(self):
        return RandomBot()
