from pinochle.Game import Game
from classes.Human import Human

player_1 = Human("Romulus")
player_2 = Human("Remus")
player_list = [player_1, player_2]
game = Game("pinochle", player_list)

print("In the red corner: " + player_1.name)
print("In the blue corner: " + player_2.name)
game.deal()
print("Trump of the round: " + game.trump)

game.play()
