from pinochle.Game import Game
from classes.Human import Human
from pinochle.State import State

player_1 = Human("Romulus")
player_2 = Human("Remus")
player_list = [player_1, player_2]
game = Game("pinochle", player_list)

# print("In the red corner: " + player_1.name)
# print("In the blue corner: " + player_2.name)
game.deal()
# print("Trump of the round: " + game.trump)

state = State(game)

state.convert_to_human_readable_format(player_1)

print(state.global_state)

# game.play()

