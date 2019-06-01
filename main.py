from classes.Suits import Suits
from classes.Value import Value
from classes.Card import Card
from util.Constants import Constants


card = Card(Constants.CLUBS, 10)

print(card.suit)
print(card.value)