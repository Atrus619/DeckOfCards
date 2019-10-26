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
from util.util import get_epsilon_constant_decrement, get_epsilon_linear_anneal
import util.util as util
from datasets.GameHistory import GameHistory
from models.DQN import DQN
from pinochle.Game import Game
from util import db
from pipeline import benchmark
from torch.utils.data import DataLoader
import logging
import time
from util.Constants import Constants as cs
import os

cfg.run_id = util.generate_run_id()
util.setup_file_logger(name=cfg.run_id, filename=cfg.run_id)
logger = logging.getLogger(cfg.run_id)
start_time = time.time()

# Define players
model_1 = DQN(**cfg.DQN_params)
model_2 = model_1.copy()

player_list = util.init_players(model_1=model_1,
                                name_1=cfg.bot_1_name,
                                model_2=model_2,
                                name_2=cfg.bot_2_name,
                                epsilon_func=globals()[cfg.epsilon_func])  # Importing string function name from config intentionally

player_1_winrate = []
previous_experience_id = 0

util.save_config(config=cfg, path=cfg.run_id)

# For each cycle
logger.info('Beginning run titled: ' + cfg.run_id)
for i in range(1, cfg.num_cycles + 1):
    winner_list = []

    # For each episode, play through episode and insert each state/action pair into the database
    logger.info('Beginning cycle: ' + str(i) + ' / ' + str(cfg.num_cycles) + '\tCumulative Time Elapsed: ' + util.get_pretty_time(time.time() - start_time))
    cycle_start_time = time.time()

    for j in range(cfg.episodes_per_cycle):
        # Initialize game
        game = Game(name=cfg.game, players=player_list, run_id=cfg.run_id, current_cycle=i)

        # Play game
        game.deal()
        winner_list.append(game.play())

    # Import data from database based on experience replay buffer
    logger.info('Data collection complete.\tTotal Episode Time: ' + util.get_pretty_time(time.time() - cycle_start_time))
    logger.info('Loading experience and training model...')
    training_start_time = time.time()

    df = db.get_exp(run_id=cfg.run_id, buffer=cfg.experience_replay_buffer)
    gh = GameHistory(df=df, **cfg.GH_params)
    gh_gen = DataLoader(dataset=gh, batch_size=gh.batch_size, shuffle=True, num_workers=cfg.num_workers)

    # Train model
    model_1.train_self(num_epochs=cfg.epochs_per_cycle, exp_gen=gh_gen, store_history=cfg.store_history)

    logger.info('Model training complete.\tTotal Training Time: ' + util.get_pretty_time(time.time() - training_start_time))

    # Update model_2
    if i % cfg.player_2_update_freq == 0:
        logger.info(cs.DIVIDER)
        logger.info('Setting model 2 equal to model 1...')
        logger.info(cs.DIVIDER)
        model_2 = model_1.copy()

    # Benchmark
    if i % cfg.benchmark_freq == 0:
        logger.info(cs.DIVIDER)
        logger.info('Benchmarking...')

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
        logger.info(cs.DIVIDER)

    # Checkpoint
    if cfg.checkpoint_freq is not None and i % cfg.checkpoint_freq == 0:
        logger.info('Model checkpoint reached. Saving checkpoint...')
        model_1.save(folder=os.path.join(cfg.checkpoint_folder, cfg.run_id), title=util.get_checkpoint_model_name(cycle=i))

    logger.info('Cycle ' + str(i) + ' / ' + str(cfg.num_cycles) + ' complete.\tTotal Cycle Time: ' + util.get_pretty_time(time.time() - cycle_start_time))
    logger.info(cs.DIVIDER)

logger.info('Training complete.\tTotal Run Time: ' + util.get_pretty_time(time.time() - start_time) + '\tSaving model and exiting...')
model_1.save(folder=cfg.saved_models_folder, title=cfg.run_id)
