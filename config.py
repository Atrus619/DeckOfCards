import logging
from pinochle.scripted_bots.RandomBot import RandomBot


class Config:
    # Meta-parameters
    num_cycles = 5
    episodes_per_cycle = 10
    experience_replay_buffer = 1e7
    epochs_per_cycle = 100
    player_2_update_freq = 10
    epsilon_schedule = None
    game = 'pinochle'
    run_id = 'FIRST_TEST'
    logging_level = logging.INFO

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
