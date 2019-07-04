class Player:
    def __init__(self, name, ai=False):
        """
        """
        self.motto = "Don't hate the player, hate the game."

        self.philosophy = "card_dict = list_to_dict(card_list)"
        """
        """

        self.name = name

        self.ai = ai

    def __str__(self):
        return self.name

