class Config:
    # Meta-parameters
    num_cycles = 50
    episodes_per_cycle = 1000
    experience_replay_buffer = 1e7
    epochs_per_cycle = 100
    player_2_update_freq = 10
    epsilon_schedule = None

    # NN Parameters
    learning_rate = None
    num_layers = None
    units_per_layer = None
    activation_func = None
    optimizer = None
    loss_metric = None
