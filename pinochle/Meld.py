from operator import itemgetter


class Meld:
    def __init__(self):
        self.melded_cards = []  # List of tuples

    def add_melded_card(self, card, combo, meld_class, score):
        self.melded_cards.append((card, combo, meld_class, score))

    def pull_melded_card(self, card_tuple):
        # card_tuple = (card, combo, meld_class, score)
        assert card_tuple in self.melded_cards, "Somehow you managed to attempt to pull an invalid card. No idea how."

        self.melded_cards.remove(card_tuple)
        return card_tuple

    def rearrange_meld(self):
        self.melded_cards = sorted(self.melded_cards, key=itemgetter(2, 3, 1))  # Sort by meld_class, then score, then combo

    def show_meld(self):
        self.rearrange_meld()

        print("Current Meld:")

        i = 0
        for card_tuple in self.melded_cards:
            i += 1
            print(str(i) + ":", ", ".join([str(val) for val in card_tuple]))

    def clear_meld(self):
        self.melded_cards = []
