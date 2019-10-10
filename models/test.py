from models.dqn import *
from datasets.GameHistory import *

model = DQN(3, 10, 2, 2)

# TODO: test out transition batching
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
