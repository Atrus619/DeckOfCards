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
import util.db as db
from config import Config as cfg
import pipeline.util as pu
from torch.utils.data import DataLoader
import pipeline.experiment as exp

util.clear_run(run_id='TEST')
exp.run_full_experiment(config=cfg)

viz.plot_diagnostic_plots(cfg.run_id)

# df = db.get_exp('TEST', 1000)
# gh = GameHistory(df=df, **cfg.GH_params)
#
# self = DQN(run_id=cfg.run_id, **cfg.DQN_params)
# exp_gen = DataLoader(dataset=gh, batch_size=gh.batch_size, shuffle=True, num_workers=cfg.num_workers)
#
# self.train_self(num_epochs=5,
#                 exp_gen=exp_gen,
#                 is_storing_history=True)

# states = data.state.to(self.device)
# actions = data.action.to(self.device)
# meld_actions = data.meld_action.to(self.device)
# next_states = data.next_state.to(self.device)
# rewards = data.reward.to(self.device)
#
# pu.train_model(model=self, config=cfg)

epsilon = Epsilon('eval')
player_1 = Agent(name=cfg.random_bot_name + '1', model=RandomBot(), epsilon=epsilon)
player_2 = Agent(name=cfg.random_bot_name + '2', model=RandomBot(), epsilon=epsilon)
player_human = Human(name='Me, no you?')

player_list = [player_1, player_2]

game = Game(name=cfg.game, players=player_list, run_id='TEST', current_cycle=None)
game.deal()
winner_index, exp_df = game.play()
db.upload_exp(df=exp_df)


import pdb; pdb.pm()

bench.benchmark_test(RandomBot(), ExpertPolicy(), 50)

no_gamma = util.get_model_checkpoint('No_Gamma')
gamma1 = util.get_model_checkpoint('Small_Gamma1')
gamma2 = util.get_model_checkpoint('Small_Gamma2')
gamma3 = util.get_model_checkpoint('Small_Gamma3')
gamma4 = util.get_model_checkpoint('Small_Gamma4')
gamma5 = util.get_model_checkpoint('Small_Gamma5')

model_list = []
for i in range(50, 500, 50):
    model_list.append(util.get_model_checkpoint('gamma65_new', i))
model_list.append(ExpertPolicy())
model_list.append(RandomBot())
bench.round_robin(model_list, 100)

latest = util.get_model_checkpoint('Long_Test')

bench.human_test(util.get_model_checkpoint('gamma85_fixed'))

g75 = util.get_model_checkpoint('gamma75', 350)
g85 = util.get_model_checkpoint('gamma85', 350)


model_list = []
for i in range(50, 500, 50):
    model_list.append(util.get_model_checkpoint('gamma85_fixed', i))
bench.round_robin(model_list, 100)

model_list = [util.get_model_checkpoint('gamma75'), util.get_model_checkpoint('gamma85_fixed'), util.get_model_checkpoint('gamma90_fixed'), util.get_model_checkpoint('Long_Test'), ExpertPolicy(), RandomBot()]

model_list = [util.get_model_checkpoint('gamma75'), util.get_model_checkpoint('gamma65_new'), util.get_model_checkpoint('gamma70_new'), util.get_model_checkpoint('gamma75_new'), util.get_model_checkpoint('Long_Test'), ExpertPolicy()]