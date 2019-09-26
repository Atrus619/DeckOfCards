from classes.Player import Player
from util.Util import print_divider

class Human(Player):
    def __init__(self, name):
        super().__init__(name)

    def get_action(self, state, msg):
        state.convert_to_human_readable_format(self)  # TODO: NAME TBD
        print_divider()
        return input(msg)
