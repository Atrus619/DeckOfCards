# Loop through number of training cycles
# Loop through number of episodes per cycle - COMPLETE
# Set up game
# Play game
# Record along the way
# Collect relevant experience for training - COMPLETE
# Train DQN or other model - COMPLETE
# Logging/Visualization/Track Progress/Benchmark

# Stretch goals:
# Parallelize
# Add complexity slowly
# Make interface more robust
# Add debug method to jump into game
# Mess with reward function
# Mess with config
# Try other algorithms!!!

# Things to log/etc.
# Average reward - COMPLETE
# Win rate versus randombot - COMPLETE
# Win rate versus older version of bot - COMPLETE
# NN weights/histograms/loss
# Overall win-rate - COMPLETE

from config import Config as cfg
from util.util import init_players, get_epsilon_constant_decrement, get_epsilon_linear_anneal, generate_run_id
from datasets.GameHistory import GameHistory
from models.DQN import DQN
from pinochle.Game import Game
from util import db
from pipeline import benchmark
from torch.utils.data import DataLoader

# Define players
model_1 = DQN(**cfg.DQN_params)
model_2 = model_1.copy()

player_list = init_players(model_1=model_1,
                           name_1=cfg.bot_1_name,
                           model_2=model_2,
                           name_2=cfg.bot_2_name,
                           epsilon_func=globals()[cfg.epsilon_func])

player_1_winrate = []
previous_experience_id = 0

cfg.run_id = generate_run_id()

# TODO: Add logging.info stuff throughout so we can monitor progress

# For each cycle
for i in range(1, cfg.num_cycles + 1):
    winner_list = []
    # For each episode, play through episode and insert each state/action pair into the database
    for j in range(cfg.episodes_per_cycle):
        # Initialize game
        game = Game(name=cfg.game, players=player_list, run_id=cfg.run_id, current_cycle=i)

        # Play game
        game.deal()
        winner_list.append(game.play())

    # Import data from database based on experience replay buffer
    df = db.get_exp(run_id=cfg.run_id, buffer=cfg.experience_replay_buffer)
    gh = GameHistory(df=df, **cfg.GH_params)
    gh_gen = DataLoader(dataset=gh, batch_size=gh.batch_size, shuffle=True, num_workers=cfg.num_workers)

    # Train model
    model_1.train_self(num_epochs=cfg.epochs_per_cycle, exp_gen=gh_gen)

    # Update model_2
    if i % cfg.player_2_update_freq == 0:
        model_2 = model_1.copy()

    # Benchmark
    if i % cfg.benchmark_freq == 0:
        # List of player 1's win rate against player 2 by cycle
        cycle_win_rate = 1 - sum(winner_list) / len(winner_list)
        player_1_winrate.append(cycle_win_rate)

        # Play against random bot and measure win rate
        model_copy = model_1.copy()
        random_win_rate = benchmark.random_bot_test(model_copy)

        # Collect average reward from database
        average_reward = benchmark.get_average_reward(cfg.run_id, previous_experience_id, cfg.bot_1_name)
        db.insert_metrics(cfg.run_id, cycle_win_rate, random_win_rate, average_reward)

        previous_experience_id = db.get_max_id(cfg.run_id)
