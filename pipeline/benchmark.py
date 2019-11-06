from config import Config as cfg
from pinochle.Game import Game
from util import db
from classes.Agent import Agent
from classes.Epsilon import Epsilon
import logging
import util.util as util
from classes.Human import Human
from util.Constants import Constants as cs

logging.basicConfig(format='%(levelname)s:%(message)s', level=cfg.logging_level)


def benchmark_test(primary_model, benchmark_model, num_games, benchmark_bot_name='benchmark_bot', run_id=None):
    winner_list = []
    epsilon = Epsilon()
    player_1 = Agent(name=cfg.bot_1_name, model=primary_model, epsilon=epsilon)
    player_2 = Agent(name=benchmark_bot_name, model=benchmark_model, epsilon=epsilon)

    if 'policy_net' in dir(player_1.model):
        player_1.model.policy_net.eval()

    if 'policy_net' in dir(player_2.model):
        player_2.model.policy_net.eval()

    for j in range(num_games):
        player_list = [player_1, player_2]
        game = Game(name="pinochle", players=player_list, run_id=run_id, current_cycle=None)
        game.deal()
        winner_list.append(game.play())

    return 1 - sum(winner_list) / len(winner_list)


def human_test(model):
    epsilon = Epsilon()
    player_1 = Agent(name=cfg.bot_1_name, model=model, epsilon=epsilon)
    player_2 = Human("Hades")
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


def round_robin(model_list, num_games, verbose=True):
    epsilon = Epsilon

    model_wins = {}
    for i, model in enumerate(model_list):
        model_wins[f'Player {i}'] = [0, model]

    for i, p1_model in enumerate(model_list):
        for j, p2_model in enumerate(model_list):
            if i < j:
                p1 = Agent(name=f'Player {i}', model=p1_model, epsilon=epsilon)
                p2 = Agent(name=f'Player {j}', model=p2_model, epsilon=epsilon)

                if verbose:
                    print(f'Player {i} vs. Player {j}...')

                p1_wins = int(benchmark_test(primary_model=p1_model, benchmark_model=p2_model, num_games=num_games) * num_games)
                p2_wins = int(num_games - p1_wins)

                if verbose:
                    print(f'Player {i}: {p1_wins}\tPlayer {j}: {p2_wins}')
                    print(cs.DIVIDER)

                model_wins[p1.name][0] += p1_wins
                model_wins[p2.name][0] += p2_wins

    output = sorted(model_wins.items(), key=lambda kv: kv[1][0], reverse=True)

    if verbose:
        for i, model in enumerate(output):
            print(f'Rank {i+1}: {model[0]} with {model[1][0]} wins')

    return output
