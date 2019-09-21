class Player:
    def __init__(self, name):
        """
        """
        self.motto = "Don't hate the player, hate the game."

        self.philosophy = "card_dict = list_to_dict(card_list)"
        """
        """

        self.name = name

    def __str__(self):
        return self.name

    def __eq__(self, other):
        return self.name == other.name

