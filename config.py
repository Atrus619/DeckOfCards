import logging
import numpy as np
import torch


class Config:
    # Meta-parameters
    num_cycles = 100
    episodes_per_cycle = 50
    experience_replay_buffer = 1e7
    epochs_per_cycle = 100
    player_2_update_freq = 10
    game = 'pinochle'
    run_id = 'TEST'
    logging_level = logging.INFO
    max_epsilon = 1.0
    min_epsilon = 0.05
    epsilon_decrement = 0.01  # for get_epsilon_constant_decrement func
    epsilon_func = 'get_epsilon_linear_anneal'
    state_size = 28
    num_actions = 24
    illegal_move_punishment = -6.9

    # NN Parameters
    DQN_params = {
        'update_target_net_freq': 10,  # In epochs
        'gamma': 0.999,  # discount rate
        'grad_clamp': True,  # Whether to clamp gradients to be between -1 and 1 (for stability purposes),
        'state_size': state_size,
        'num_actions': num_actions,
        'terminal_state_tensor': torch.zeros(state_size),
        'num_layers': 3,
        'hidden_units_per_layer': 10,
        'device': 'cpu'
    }

    # Custom Data Set Parameters
    GH_params = {
        'batch_size': 1024,
        'state_size': state_size,
        'num_actions': num_actions,
        'device': 'cpu'
    }

    # Agent Parameters
    bot_1_name = 'xXxPussySlayer69xXx'
    bot_2_name = '007'

    # Benchmark Parameters
    benchmark_freq = 5
    random_bot_name = 'RANDOM_BOT_TEST'
    random_bot_cycles = 30

    # Needs to match length of state_vector
    terminal_state = ','.join([str(x) for x in np.zeros(state_size)])

    # Misc Parameters
    num_workers = 6  # for the data loader to feed the training of NN
