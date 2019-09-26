import numpy as np
from config import Config as cfg
from pinochle.Game import Game
from classes.Agent import Agent
from pinochle.scripted_bots.RandomBot import RandomBot


class DQN:
    """
    deep Q learning network
    """

    def __init__(self):
        pass

    def train(self, num_epoch):
        pass

    def train_one_epoch(self):
        for i in range(cfg.episodes):
            bot1 = RandomBot(24)
            player_1 = Agent(name="700", model=bot1)
            bot1.assign_player(player_1)

            bot2 = RandomBot(24)
            player_2 = Agent(name="007", model=bot2)
            bot2.assign_player(player_2)

            player_list = [player_1, player_2]
            game = Game("pinochle", player_list, "1")

            game.play()



