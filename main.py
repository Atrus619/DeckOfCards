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
import pipeline.benchmark as bench

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

bench.benchmark_test(RandomBot(), ExpertPolicy(), 'Leon', 30)

no_gamma = util.get_model_checkpoint('No_Gamma')
gamma1 = util.get_model_checkpoint('Small_Gamma1')
gamma2 = util.get_model_checkpoint('Small_Gamma2')
gamma3 = util.get_model_checkpoint('Small_Gamma3')
gamma4 = util.get_model_checkpoint('Small_Gamma4')
gamma5 = util.get_model_checkpoint('Small_Gamma5')

gamma4_50 = util.get_model_checkpoint('Small_Gamma4', 50)
gamma4_100 = util.get_model_checkpoint('Small_Gamma4', 100)
gamma4_150 = util.get_model_checkpoint('Small_Gamma4', 150)
gamma4_200 = util.get_model_checkpoint('Small_Gamma4', 200)

model_list = [gamma4_50, gamma4_100, gamma4_150, gamma4_200]
model_list = [no_gamma, gamma1, gamma2, gamma3, gamma4, gamma5]
bench.round_robin(model_list, 100)

bench.benchmark_test(gamma4_100, gamma4_200, 'test', 150)

bench.human_test(gamma4_50)