from config import Config as cfg
from util.util import init_players
from pinochle.scripted_bots.RandomBot import RandomBot
from pinochle.Game import Game
from util import db
from classes.Agent import Agent
import logging

logging.basicConfig(format='%(levelname)s:%(message)s', level=cfg.logging_level)


def random_bot_test(model):
    winner_list = []
    random_bot = RandomBot()
    player_2 = Agent(name=cfg.random_bot_name, model=random_bot)
    random_bot.assign_player(player_2)

    for j in range(cfg.random_bot_cycles):
        # Initialize game
        player_list = [model.player, player_2]
        game = Game("pinochle", player_list, None)
        game.deal()
        winner_list.append(game.play())

    return 1 - sum(winner_list) / len(winner_list)
