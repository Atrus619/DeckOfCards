from classes.Player import Player


class Human(Player):
    def __init__(self, name):
        super().__init__(name)

    def get_action(self, state, msg):
        state.convert_to_human_readable_format(self)  # TODO: NAME TBD
        return input(msg)
