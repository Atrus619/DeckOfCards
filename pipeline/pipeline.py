# Loop through number of training cycles
    # Loop through number of episodes per cycle
        # Set up game
        # Play game
        # Record along the way
    # Collect relevant experience for training
    # Train DQN or other model
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
    # Average reward
    # Win rate versus randombot
    # Win rate versus older version of bot
    # NN weights/histograms/loss
    # Overall win-rate

from config import Config as cfg
from util.util import init_players
from pinochle.scripted_bots.RandomBot import RandomBot
from pinochle.Game import Game
from util import db
from pipeline.benchmark import random_bot_test

# Define players
model_1 = RandomBot()
model_2 = model_1.copy()

player_list = init_players(model_1=model_1,
                           name_1=cfg.bot_1_name,
                           model_2=model_2,
                           name_2=cfg.bot_2_name)

player_1_winrate = []

# For each cycle
for i in range(cfg.num_cycles):
    winner_list = []
    # For each episode
    for j in range(cfg.episodes_per_cycle):
        # Initialize game
        game = Game(name=cfg.game, players=player_list, run_id=cfg.run_id)
        # Play game
        game.deal()
        winner_list.append(game.play())

    # Collect data based on experience replay buffer
    data = db.get_exp(run_id=cfg.run_id, buffer=cfg.experience_replay_buffer)

    # Train model here
    model_1.train(num_epochs=cfg.epochs_per_cycle)

    # Update model_2
    if i % cfg.player_2_update_freq == 0:
        model_2 = model_1.copy()

    # Benchmark and stuff
    if i % cfg.benchmark_freq == 0:
        # List of player 1's winrate against player 2 by cycle
        cycle_win_rate = 1 - sum(winner_list)/len(winner_list)
        player_1_winrate.append(cycle_win_rate)

        print(model_1)
        print(model_1.player.name)
        random_win_rate = random_bot_test(model_1)

        #TODO: figure out how to calculate average reward
        db.insert_win_rate(cfg.run_id, cycle_win_rate, random_win_rate, 0)

