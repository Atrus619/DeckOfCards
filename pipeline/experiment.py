# Loop through number of training cycles - COMPLETE
# Loop through number of episodes per cycle - COMPLETE
# Set up game - COMPLETE
# Play game - COMPLETE
# Record along the way - COMPLETE
# Collect relevant experience for training - COMPLETE
# Train DQN or other model - COMPLETE
# Logging/Visualization/Track Progress/Benchmark - COMPLETE

# Stretch goals:
# Parallelize
# Add complexity slowly
# Make interface more robust
# Add debug method to jump into game
# Mess with reward function - COMPLETE
# Mess with config - COMPLETE
# Try other algorithms!!!

# Things to log/etc.
# Average reward - COMPLETE
# Win rate versus randombot - COMPLETE
# Win rate versus older version of bot - COMPLETE
# NN weights/histograms/loss - COMPLETE
# Overall win-rate - COMPLETE

import util.util as util
from datasets.GameHistory import GameHistory
from models.DQN import DQN
from pinochle.Game import Game
from classes.Epsilon import Epsilon
from classes.Agent import Agent
from util import db
from pipeline import benchmark
from torch.utils.data import DataLoader
import logging
import time
from util.Constants import Constants as cs
from pinochle.scripted_bots.RandomBot import RandomBot
from pinochle.scripted_bots.ExpertPolicy import ExpertPolicy
import os


def run_full_experiment(config):
    util.setup_file_logger(name=config.run_id, filename=config.run_id)
    logger = logging.getLogger(config.run_id)
    start_time = time.time()
    
    # Define players
    model_1 = DQN(**config.DQN_params)
    model_2 = model_1.copy()
    epsilon = Epsilon(epsilon_func=config.epsilon_func, max_epsilon=config.max_epsilon, min_epsilon=config.min_epsilon,
                      eval_epsilon=config.eval_epsilon, num_cycles=config.num_cycles, decrement=config.epsilon_decrement)

    player_list = [Agent(name=config.bot_1_name, model=model_1, epsilon=epsilon), Agent(name=config.bot_2_name, model=model_2, epsilon=epsilon)]

    player_1_winrate = []
    previous_experience_id = 0
    
    util.save_config(config=config, path=config.run_id)
    
    # For each cycle
    logger.info('Beginning run titled: ' + config.run_id)
    logger.info(cs.DIVIDER)

    for i in range(1, config.num_cycles + 1):
        winner_list = []
    
        # For each episode, play through episode and insert each state/action pair into the database
        logger.info('Beginning cycle: ' + str(i) + ' / ' + str(config.num_cycles) + '\tCumulative Time Elapsed: ' + util.get_pretty_time(time.time() - start_time))
        logger.info(f'Current Epsilon: {epsilon.get_epsilon(current_cycle=i):.3f}')
        cycle_start_time = time.time()
    
        for j in range(config.episodes_per_cycle):
            # Initialize game
            game = Game(name=config.game, players=player_list, run_id=config.run_id, current_cycle=i)
    
            # Play game
            game.deal()
            winner_list.append(game.play())
    
        # Import data from database based on experience replay buffer
        logger.info('Data collection complete.\tTotal Episode Time: ' + util.get_pretty_time(time.time() - cycle_start_time))
        logger.info('Loading experience and training model...')
        training_start_time = time.time()
    
        df = db.get_exp(run_id=config.run_id, buffer=config.experience_replay_buffer)
        gh = GameHistory(df=df, **config.GH_params)
        gh_gen = DataLoader(dataset=gh, batch_size=gh.batch_size, shuffle=True, num_workers=config.num_workers)
    
        # Train model
        model_1.train_self(num_epochs=config.epochs_per_cycle, exp_gen=gh_gen, store_history=config.store_history)
    
        logger.info('Model training complete.\tTotal Training Time: ' + util.get_pretty_time(time.time() - training_start_time))
    
        # Update model_2
        if i % config.player_2_update_freq == 0:
            logger.info(cs.DIVIDER)
            logger.info('Setting model 2 equal to model 1...')
            player_list[1].set_model(model=model_1.copy())
    
        # Benchmark
        if i % config.benchmark_freq == 0:
            logger.info(cs.DIVIDER)
            logger.info('Benchmarking...')
    
            # List of player 1's win rate against player 2 by cycle
            cycle_win_rate = 1 - sum(winner_list) / len(winner_list)
            player_1_winrate.append(cycle_win_rate)
    
            # Play against random bot and measure win rate
            random_win_rate = benchmark.benchmark_test(model=model_1, benchmark_model=RandomBot(), benchmark_bot_name=config.random_bot_name,
                                                       run_id=config.run_id if config.log_random_benchmark else None)
            logger.info(f'Winrate vs. Random Bot: {random_win_rate * 100:.1f}%')

            # Play against expert policy bot and measure win rate
            expert_policy_win_rate = benchmark.benchmark_test(model=model_1, benchmark_model=ExpertPolicy(), benchmark_bot_name=config.expert_policy_bot_name,
                                                              run_id=config.run_id if config.log_expert_policy_benchmark else None)
            logger.info(f'Winrate vs. Expert Policy: {expert_policy_win_rate * 100:.1f}%')
    
            # Collect average reward from database
            average_reward = benchmark.get_average_reward(run_id=config.run_id, previous_experience_id=previous_experience_id,
                                                          agent_id=config.bot_1_name, opponent_id=config.bot_2_name)
            db.insert_metrics(run_id=config.run_id, win_rate=cycle_win_rate, win_rate_random=random_win_rate, win_rate_expert_policy=expert_policy_win_rate,
                              average_reward=average_reward)
    
            previous_experience_id = db.get_max_id(config.run_id)

        # Checkpoint
        if config.checkpoint_freq is not None and i % config.checkpoint_freq == 0:
            logger.info(cs.DIVIDER)
            logger.info('Model checkpoint reached. Saving checkpoint...')
            model_1.save(folder=os.path.join(config.checkpoint_folder, config.run_id), title=util.get_checkpoint_model_name(cycle=i))

        logger.info('Cycle complete.\tTotal Cycle Time: ' + util.get_pretty_time(time.time() - cycle_start_time))
        logger.info(cs.DIVIDER)

    logging.info('Training complete.\tTotal Run Time: ' + util.get_pretty_time(time.time() - start_time) + '\tSaving model and exiting...')
    model_1.save(title=config.run_id)
