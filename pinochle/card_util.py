from util.Constants import Constants as cs


def compare_cards(trump, card_1, card_2):
    """
    :param trump
    :param card_1:
    :param card_2:
    :return: the number that represents the card that won
    """
    value_map = cs.STANDARD_CARDS_VALUE

    if card_1 == card_2:
        return 0

    if trump == card_1.suit:
        if trump != card_2.suit:
            return 0
        elif value_map[card_1.value] >= value_map[card_2.value]:
            return 0
        else:
            return 1

    elif trump == card_2.suit:
        return 1

    elif card_1.suit == card_2.suit:
        if value_map[card_1.value] >= value_map[card_2.value]:
            return 0
        else:
            return 1

    else:
        return 0
