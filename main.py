from pinochle.Game import Game
from classes.Human import Human
from classes.Agent import Agent
from pinochle.scripted_bots.RandomBot import RandomBot
from pinochle.State import State
from util import util
from datasets.GameHistory import GameHistory
from torch.utils import data
import numpy as np
import torch
from config import Config as cfg
from models.DQN import DQN
from pinochle.scripted_bots.TheProfessional import TheProfessional


player_1 = Agent(name="Leon", model=TheProfessional(), epsilon_func=util.get_expert_epsilon)
player_2 = Agent(name=cfg.random_bot_name, model=RandomBot(), epsilon_func=util.get_random_bot_epsilon)

# Initialize game
player_list = [player_1, player_2]
game = Game(name="pinochle", players=player_list, run_id=None, current_cycle=None)
game.deal()
game.play()

