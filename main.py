from classes.Suits import Suits
from classes.Value import Value
from classes.Card import Card
from classes.Deck import Deck
from util.Constants import Constants as cs
from classes.Hand import Hand


card = Card(cs.TEN, cs.SPADES)

print(card.suit)
print(card.value)

hand = Hand()
deck = Deck()
card1 = Card('two', 'hearts')
card2 = Card('three', 'hearts')
card3 = Card('two', 'hearts')

card_list = [card1, card2, card3]

hand.draw(card_list)
hand2 = Hand()
hand2.draw(card_list)
