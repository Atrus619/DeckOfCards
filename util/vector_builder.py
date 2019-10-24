import numpy as np
from operator import attrgetter
from util.Vectors import Vectors as vs


def build_hand_vector(hand):
    """
    Produce a one hot vector based on the deck template for a player's hand
    :param hand: List of cards
    :return: NumPy Array
    """

    hand = sorted(hand, key=attrgetter('suit', 'numeric_value'))
    output = np.zeros(len(vs.PINOCHLE_ONE_HOT_VECTOR))
    last_index = 0
    for hand_index, hand_card in enumerate(hand):
        if hand_index == 0 or hand_card != hand[hand_index - 1]:
            for template_index, template_card in enumerate(vs.PINOCHLE_ONE_HOT_VECTOR):
                if hand_card == template_card:
                    last_index = template_index
                    output[last_index] = 1
                    break
        else:
            output[last_index] = 2

    return output


def build_trump_vector(trump):
    """
    Produce one hot vector based on the trump card in the current game
    :return:
    """
    suit_template = sorted(set([card.suit for card in vs.PINOCHLE_ONE_HOT_VECTOR]))  # TODO: Extremely overkill, might want to hardcode for performance gainz
    output = np.zeros(len(suit_template))

    for index, suit in enumerate(suit_template):
        if suit == trump:
            output[index] = 1
            return output


def build_card_vector(card):
    """
    Produce a one hot vector based on the deck template for a card
    Can handle no card submission, needed for first played card vector
    :param card: card
    :return: NumPy Array
    """

    output = np.zeros(len(vs.PINOCHLE_ONE_HOT_VECTOR))
    if card is None:
        return output

    for template_index, template_card in enumerate(vs.PINOCHLE_ONE_HOT_VECTOR):
        if card == template_card:
            last_index = template_index
            output[last_index] = 1
            return output


def build_meld_cards_vector(mt_list):
    """
    Produce a one hot vector based on the deck template for a player's meld cards
    :param mt_list: List of meld tuples
    :return: NumPy Array
    """

    output = np.zeros(len(vs.PINOCHLE_ONE_HOT_VECTOR))
    for meld_tuple in mt_list:
        output += build_card_vector(meld_tuple[0].card)

    return output