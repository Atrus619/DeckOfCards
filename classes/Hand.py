class Hand:
    def __init__(self):
        self.cards = []

    def __getitem__(self, item):
        return self.cards[item]

    def __len__(self):
        return len(self.cards)

    def __eq__(self, other):
        # TODO: Feel free to optimize this. I was bored. Sorting would be ideal...but sounds complicated...USE DICTIONARIES
        if not isinstance(other, self.__class__):
            return False
        elif len(self.cards) != len(other.cards):
            return False
        elif not all([x in other.cards for x in self.cards]):
            return False
        elif not all([x in self.cards for x in other.cards]):
            return False
        else:
            return True

    def add_cards(self, cards_list):
        """
        Adds cards to hand.
        :param cards_list: List of cards to add.
        :return: Nothing.
        """
        if type(cards_list) != list:  # In case a single card is drawn that someone doesn't put into a list
            cards_list = [cards_list]
        for card in cards_list:
            self.cards.append(card)

    def drop_cards(self, cards_list):
        """
        Removes cards from hand.
        :param cards_list: List of cards to remove.
        :return: True if successful, false if unsuccessful (attempting to remove cards that do not exist in hand).
        """
        if type(cards_list) != list:  # In case a single card is drawn that someone doesn't put into a list
            cards_list = [cards_list]
        for card in cards_list:
            if card in self.cards:
                self.cards.remove(card)
            else:
                return False
        return True

    def show(self):
        i = 0
        for card in self.cards:
            i += 1
            print(str(i) + ":", card)

    def clear(self):
        self.cards = []

    def pull_card(self, card):
        assert card in self.cards

        self.cards.remove(card)
        return card

