import logging
import numpy as np
import torch


class Config:
    # Meta-parameters
    num_cycles = 100
    episodes_per_cycle = 50
    experience_replay_buffer = 1e5
    epochs_per_cycle = 20
    player_2_update_freq = 10
    game = 'pinochle'
    run_id = 'TEST'
    logging_level = logging.INFO
    store_history = True  # Whether to store gradients & friends in the neural network class
    checkpoint_freq = 5  # Frequency (in cycles) to checkpoint model (save to checkpoints folder)

    # Epsilon
    epsilon_func = 'get_epsilon_linear_anneal'
    max_epsilon = 1.0
    min_epsilon = 0.05
    eval_epsilon = 0.0
    epsilon_decrement = 0.01  # for get_epsilon_constant_decrement func

    # State-Action
    state_size = 76
    num_actions = 24

    # Benchmark Parameters
    benchmark_freq = 5
    random_bot_name = 'RANDOM_BOT_TEST'
    random_bot_cycles = 30

    # Human benchmark
    human_test = False

    # NN Parameters
    DQN_params = {
        'update_target_net_freq': 10,  # In epochs
        'gamma': 0.999,  # discount rate
        'grad_clamp': True,  # Whether to clamp gradients to be between -1 and 1 (for stability purposes),
        'state_size': state_size,
        'num_actions': num_actions,
        'terminal_state_tensor': torch.zeros(state_size),
        'num_layers': 3,
        'hidden_units_per_layer': 32,
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

    # Needs to match length of state_vector
    terminal_state = ','.join([str(x) for x in np.zeros(state_size)])

    # Misc Parameters
    num_workers = 0  # for the data loader to feed the training of NN
    log_folder = 'logs'
    saved_models_folder = 'saved_models'
    checkpoint_folder = 'model_checkpoints'
