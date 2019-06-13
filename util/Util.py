def list_to_dict(card_list):
    card_dict = {}

    for card in card_list:

        if card not in card_dict:
            card_dict[card] = 1
        else:
            card_dict[card] += 1

    return card_dict

