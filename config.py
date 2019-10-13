import logging
from pinochle.scripted_bots.RandomBot import RandomBot
import numpy as np
import torch


class Config:
    # Meta-parameters
    num_cycles = 5
    episodes_per_cycle = 10
    experience_replay_buffer = 1e7
    epochs_per_cycle = 100
    player_2_update_freq = 10
    epsilon_schedule = None
    game = 'pinochle'
    run_id = 'TEST'
    logging_level = logging.INFO
    state_size = 28
    num_actions = 24
    update_target_net_freq = 10  # In epochs
    gamma = 0.999  # Discount rate
    grad_clamp = True  # Whether to clamp gradients to be between -1 and 1 (for stability purposes)

    # NN Parameters
    learning_rate = None
    num_layers = None
    units_per_layer = None
    activation_func = None
    optimizer = None
    loss_metric = None

    # Agent Parameters
    bot_1_name = 'xXxPussySlayer69xXx'
    bot_2_name = '007'

    # Benchmark Parameters
    benchmark_freq = 1
    random_bot_name = 'RANDOM_BOT_TEST'
    random_bot_cycles = 5

    # Needs to match length of state_vector
    terminal_state = ','.join([str(x) for x in np.zeros(state_size)])
    terminal_state_tensor = torch.zeros(state_size)

    # Misc Parameters
    num_workers = 4  # for the data loader to feed the training of NN
