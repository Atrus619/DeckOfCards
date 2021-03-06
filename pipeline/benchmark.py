from config import Config as cfg
from pinochle.Game import Game
from util import db
from classes.Agent import Agent
from classes.Epsilon import Epsilon
import logging
import util.util as util
from classes.Human import Human
from util.Constants import Constants as cs
import matplotlib.pyplot as plt
from collections import OrderedDict
import time
import pipeline.util as pu

logging.basicConfig(format='%(levelname)s:%(message)s', level=cfg.logging_level)


def benchmark_test(primary_model, benchmark_model, num_games, benchmark_bot_name='benchmark_bot', run_id=None):
    epsilon = Epsilon()
    player_1 = Agent(name=cfg.bot_1_name, model=primary_model, epsilon=epsilon)
    player_2 = Agent(name=benchmark_bot_name, model=benchmark_model, epsilon=epsilon)

    if 'policy_net' in dir(player_1.model):
        player_1.model.policy_net.eval()

    if 'policy_net' in dir(player_2.model):
        player_2.model.policy_net.eval()

    game_output = []
    for j in range(num_games):
        player_list = [player_1, player_2]
        game = Game(name="pinochle", players=player_list, run_id=run_id, current_cycle=None)
        game.deal()
        game_output.append(game.play())

    winner_list, exp_df = pu.parse_game_output(game_output=game_output)

    if run_id is not None:  # Store history
        db.upload_exp(df=exp_df)

    return 1 - sum(winner_list) / len(winner_list)


def human_test(model):
    epsilon = Epsilon()
    player_1 = Agent(name=cfg.bot_1_name, model=model, epsilon=epsilon)
    player_2 = Human("YOU")

    if 'policy_net' in dir(model):
        model.policy_net.eval()

    # Set logging level to debug
    logging.getLogger().setLevel(logging.DEBUG)
    logging.info("Human test enabled, initializing AI uprising...")

    # Initialize game
    player_list = [player_1, player_2]
    game = Game(name="pinochle", players=player_list, run_id=None, current_cycle=None, human_test=True)
    game.deal()
    game.play()

    # Set logging level back to config
    logging.getLogger().setLevel(cfg.logging_level)


def get_average_reward(run_id, previous_experience_id, agent_id, opponent_id):
    df = db.get_rewards_by_id(run_id=run_id, previous_experience_id=previous_experience_id, agent_id=agent_id, opponent_id=opponent_id)
    logging.debug(cfg.benchmark_freq * cfg.episodes_per_cycle)
    logging.debug(df.sum())
    average = df.sum() / (cfg.benchmark_freq * cfg.episodes_per_cycle)

    logging.info("Average reward since last benchmark from self-play: " + str(round(average.reward, 2)))

    return average.reward


def round_robin(model_list, num_games, verbose=True, plot=True, device='cuda:0'):
    start_time = time.time()
    epsilon = Epsilon

    model_wins = OrderedDict()
    for i, model in enumerate(model_list):
        model_wins[f'Player {i}'] = [0, model]
        if 'device' in dir(model) and model.device != device:
            model.policy_net = model.policy_net.to(device)

    for i, p1_model in enumerate(model_list):
        for j, p2_model in enumerate(model_list):
            if i < j:
                round_start_time = time.time()

                p1 = Agent(name=f'Player {i}', model=p1_model, epsilon=epsilon)
                p2 = Agent(name=f'Player {j}', model=p2_model, epsilon=epsilon)

                if verbose:
                    print(f'Player {i} vs. Player {j}...')

                p1_wins = int(benchmark_test(primary_model=p1_model, benchmark_model=p2_model, num_games=num_games) * num_games)
                p2_wins = int(num_games - p1_wins)

                if verbose:
                    print(f'Player {i}: {p1_wins}\tPlayer {j}: {p2_wins}\tDuration: {util.get_pretty_time(time.time() - round_start_time)}')
                    print(cs.DIVIDER)

                model_wins[p1.name][0] += p1_wins
                model_wins[p2.name][0] += p2_wins

    output = sorted(model_wins.items(), key=lambda kv: kv[1][0], reverse=True)

    if verbose:
        for i, model in enumerate(output):
            print(f'Rank {i+1}: {model[0]} with {model[1][0]} wins')
        total_games = len(model_list) / 2 * (len(model_list) - 1) * num_games
        total_duration = time.time() - start_time
        avg_time_per_game = total_duration / total_games
        print(f'{total_games} total games played over {util.get_pretty_time(total_duration)} ({util.get_pretty_time(avg_time_per_game)} per game)')

    if plot:
        xs = [x[0] for x in model_wins.items()]
        heights = [x[1][0] for x in model_wins.items()]
        plt.bar(height=heights, x=xs)
        plt.title('Round Robin Tournament Results')
        plt.xlabel('Model')
        plt.ylabel('Total Number of Wins')

    return output
