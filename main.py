from Pinochle.Game import Game
from classes.Player import Player
from Pinochle.Trick import Trick

player_1 = Player("Romulus")
player_2 = Player("Remus")
player_list = [player_1, player_2]
game = Game("pinochle", player_list)

print("In the red corner: " + player_1.name)
print("In the blue corner: " + player_2.name)
game.deal()
print("Trump of the round: " + game.trump)

trick = Trick(player_list, game.trump)
card_1 = game.hands[player_1].pull_card(game.hands[player_1][0])
card_2 = game.hands[player_2].pull_card(game.hands[player_2][0])

print("LETS GET READY TO RUMBLE!!!!!!!!!!!!!!!!!!!!!!!")
print("Card 1: " + str(card_1))
print("Card 2: " + str(card_2))

result = trick.compare_cards(card_1, card_2)

print("VICTOR : " + str(result))

