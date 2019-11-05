from pinochle.Game import Game
from classes.Human import Human
from classes.Agent import Agent
from classes.Epsilon import Epsilon
from pinochle.scripted_bots.RandomBot import RandomBot
from pinochle.State import State
from util import util
from datasets.GameHistory import GameHistory
from torch.utils import data
import numpy as np
import torch
from config import Config as cfg
from models.DQN import DQN
from pinochle.scripted_bots.ExpertPolicy import ExpertPolicy
import logging
import visualize.visualize as viz

import pipeline.experiment as exp

# exp.run_full_experiment(config=cfg)
#
# viz.plot_diagnostic_plots('No_Gamma')

epsilon = Epsilon('eval')
player_1 = Agent(name=cfg.expert_policy_bot_name, model=ExpertPolicy(), epsilon=epsilon)
player_2 = Agent(name=cfg.random_bot_name, model=RandomBot(), epsilon=epsilon)

player_list = [player_1, player_2]


game = Game(name=cfg.game, players=player_list, run_id='TEST', current_cycle=None)
game.deal()
game.play()
