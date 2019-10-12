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

exp = db.get_exp('TEST_None', 1000)
my_data = GameHistory(exp, 64, 28, 24)

my_data_gen = data.DataLoader(my_data, batch_size=my_data.bs, shuffle=True, num_workers=4)

for batch in my_data_gen:
    print("hello")

arr = exp.state.values
arr2 = arr.split(',')

test = exp.state.str.split(',')

test_list = list(map(float, test))

pls_work = test.apply(lambda x: list(map(float, x)))
