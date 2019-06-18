from pinochle.Game import Game
from classes.Player import Player
from pinochle.Trick import Trick
from classes.Card import Card
from pinochle.Meld import Meld
from util.Util import *
from util.Constants import Constants as cs


player_1 = Player("Romulus")
player_2 = Player("Remus")
player_list = [player_1, player_2]
game = Game("pinochle", player_list)

print("In the red corner: " + player_1.name)
print("In the blue corner: " + player_2.name)
game.deal()
print("Trump of the round: " + game.trump)

game.play_trick(0)

# trick = Trick(player_list, game.trump)
# card_1 = game.hands[player_1].pull_card(game.hands[player_1][0])
# card_2 = game.hands[player_2].pull_card(game.hands[player_2][0])
#
# print("LETS GET READY TO RUMBLE!!!!!!!!!!!!!!!!!!!!!!!")
# print("Card 1: " + str(card_1))
# print("Card 2: " + str(card_2))
#
# result = trick.compare_cards(card_1, card_2)
#
# print("VICTOR : " + str(result))
#
# test_card_list = [Card(cs.QUEEN, cs.SPADES), Card(cs.JACK, cs.DIAMONDS)]
#
# meld = Meld(cs.CLUBS)
#
# print("COMBO SCORE : " + str(meld.calculate_score(test_card_list)))

# test_card_list = [Card(cs.QUEEN, cs.SPADES), Card(cs.KING, cs.SPADES)]
# score = game.meld.calculate_score(test_card_list)
#
# print("COMBO SCORE : " + str(score))
