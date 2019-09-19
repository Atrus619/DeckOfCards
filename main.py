from pinochle.Game import Game
from classes.Human import Human
from classes.Agent import Agent
from pinochle.scripted_bots.RandomBot import RandomBot
from pinochle.State import State
from util import db

player_1 = Human("Romulus")

bot = RandomBot(24)
player_2 = Agent(name="007", model=bot)
bot.assign_player(player_2)

player_list = [player_1, player_2]
game = Game("pinochle", player_list)

# print("In the red corner: " + player_1.name)
# print("In the blue corner: " + player_2.name)
# game.deal()
# # print("Trump of the round: " + game.trump)
#
# state = State(game)
#
# state.convert_to_human_readable_format(player_1)
#
# print(state.global_state)
#
# game.play()

db.get_exp()

db.get_exp()

db.get_exp()

db.get_exp()