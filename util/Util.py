def list_to_dict(card_list):
    card_dict = {}

    for card in card_list:
        key = str(card)

        if key not in card_dict:
            card_dict[key] = 1
        else:
            card_dict[key] += 1

    return card_dict

