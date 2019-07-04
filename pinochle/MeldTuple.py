# TODO: Convert tuple to a class: (card, combo_name, meld_class, meld_score)
class MeldTuple:
    def __init__(self, card, combo, meld_class, score):
        self.card = card
        self.combo = combo
        self.meld_class = meld_class
        self.score = score

    def __eq__(self, other):
        return (self.card, self.combo, self.meld_class, self.score) == (other.card, other.combo, other.meld_class, other.score)
