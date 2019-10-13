from pinochle.Game import Game
from classes.Human import Human
from classes.Agent import Agent
from pinochle.scripted_bots.RandomBot import RandomBot
from pinochle.State import State
from util import db
from datasets.GameHistory import GameHistory
from torch.utils import data
import numpy as np
import torch
from config import Config as cfg
from models.DQN import DQN

exp = db.get_exp('TEST_None', 1000)
my_data = GameHistory(exp, 64, 28, 24, device='cpu')

my_data_gen = data.DataLoader(my_data, batch_size=my_data.bs, shuffle=True, num_workers=cfg.num_workers)

model = DQN(update_target_net_freq=cfg.update_target_net_freq, gamma=cfg.gamma, grad_clamp=cfg.grad_clamp, terminal_state_tensor=cfg.terminal_state_tensor,
            num_layers=3, hidden_units_per_layer=10, state_size=cfg.state_size, num_actions=cfg.num_actions)

for states, actions, next_states, rewards in my_data_gen:
    states, actions, next_states, rewards = states.to(model.device), actions.to(model.device), next_states.to(model.device), rewards.to(model.device)
    break

model.update_target_net()

model.policy_net(states.to(model.device)).gather(1, actions.to(model.device).argmax(dim=1).view(-1, 1))
test = model.policy_net(states.to(model.device))
test2 = test.gather(1, actions.to(model.device))

