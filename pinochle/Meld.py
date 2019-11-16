from operator import attrgetter
import logging
from config import Config as cfg

logging.basicConfig(format='%(levelname)s:%(message)s', level=cfg.logging_level)


class Meld:
    # mt represents a MeldTuple (see class in MeldTuple.py)
    def __init__(self):
        self.melded_cards = []  # List of tuples

    def add_melded_card(self, mt):
        self.melded_cards.append(mt)

    def pull_melded_card(self, mt):
        # card_tuple = (card, combo, meld_class, score)
        assert mt in self.melded_cards, "Somehow you managed to attempt to pull an invalid card. No idea how."

        self.melded_cards.remove(mt)
        return mt

    def rearrange_meld(self):
        # Sort by meld_class, then score, then combo
        self.melded_cards = sorted(self.melded_cards, key=attrgetter('meld_class', 'score', 'combo'))

    def show(self):
        self.rearrange_meld()

        logging.debug("Current Meld:")

        i = 0
        for mt in self.melded_cards:
            logging.debug(str(i) + ": " + str(mt.card) + ", " + mt.combo + ", " + mt.meld_class + ", " + str(mt.score))
            i += 1

    def clear_meld(self):
        self.melded_cards = []

    def is_card_present(self, card):
        for mt in self.melded_cards:
            if card == mt.card:
                return mt

        return
