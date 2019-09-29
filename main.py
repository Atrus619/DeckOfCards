from pinochle.Game import Game
from classes.Human import Human
from classes.Agent import Agent
from pinochle.scripted_bots.RandomBot import RandomBot
from pinochle.State import State
from util import db
import logging
from config import Config as cfg

logging.basicConfig(format='%(levelname)s:%(message)s', level=cfg.logging_level)

# player_1 = Human("Romulus")
bot1 = RandomBot()
player_1 = Agent(name="xXxPussySlayer69xXx", model=bot1)
bot2 = RandomBot()
player_2 = Agent(name="007", model=bot2)
bot1.assign_player(player_1)
bot2.assign_player(player_2)

player_list = [player_1, player_2]
game = Game("pinochle", player_list)

logging.debug("In the red corner: " + player_1.name + " (" + type(player_1).__name__ + ")")
logging.debug("In the blue corner: " + player_2.name + " (" + type(player_2).__name__ + ")")
game.deal()
logging.debug("Trump of the round: " + game.trump)

# state = State(game)
#
# state.convert_to_human_readable_format(player_1)
#
# logging.debug(state.global_state)

game.play()
