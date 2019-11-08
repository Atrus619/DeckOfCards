import logging
import numpy as np
import torch


class Config:
    # Meta-parameters
    num_cycles = 400
    episodes_per_cycle = 50
    experience_replay_buffer = 1e5
    epochs_per_cycle = 20
    player_2_update_freq = 10
    game = 'pinochle'
    run_id = 'TEST'
    logging_level = logging.INFO
    store_history = True  # Whether to store gradients & friends in the neural network class
    checkpoint_freq = 50  # Frequency (in cycles) to checkpoint model (save to checkpoints folder)
    win_reward = 100

    # Epsilon
    epsilon_func = 'constant_decrement'  # Valid options are 'linear_anneal', 'constant_decrement', and 'eval'
    max_epsilon = 1.0
    min_epsilon = 0.1
    eval_epsilon = 0.0
    epsilon_decrement = 0.005  # for get_epsilon_constant_decrement func

    # State-Action
    state_size = 77  # Length of state vector: [hand=24, score_diff_vector=1, trump_vector=4, discard_vector=24, played_card_vector=24]
    num_actions = 24  # Length of action vector

    # Benchmark Parameters
    benchmark_freq = 25  # How often to benchmark (in number of cycles)
    random_bot_cycles = 30  # Number of games to play vs random bot when benchmarking
    log_random_benchmark = False  # Whether to include random bot benchmarks in training data
    expert_policy_bot_cycles = random_bot_cycles  # Number of games to play vs expert policy bot when benchmarking
    log_expert_policy_benchmark = True  # Whether to include expert policy bot benchmarks in training data

    # NN Parameters
    DQN_params = {
        'update_target_net_freq': 10,  # In epochs
        'gamma': 0.85,  # discount rate
        'grad_clamp': True,  # Whether to clamp gradients to be between -1 and 1 (for stability purposes),
        'state_size': state_size,
        'num_actions': num_actions,
        'terminal_state_tensor': torch.zeros(state_size),
        'num_layers': 6,
        'hidden_units_per_layer': 32,
        'device': 'cuda:0'  # Valid choices are 'cpu' or 'cuda:0'
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
    random_bot_name = 'RANDOM_BOT_TEST'
    expert_policy_bot_name = 'Leon'

    # Needs to match length of state_vector
    terminal_state = ','.join([str(x) for x in np.zeros(state_size)])

    # Misc Parameters
    num_workers = 0  # for the data loader to feed the training of NN
    log_folder = 'logs'
    final_models_folder = 'saved_models'
    checkpoint_folder = 'model_checkpoints'
    experiment_folder = 'experiments'
    config_folder = 'saved_configs'
    human_test_pause_length = 3
