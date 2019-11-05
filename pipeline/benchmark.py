from config import Config as cfg
from pinochle.Game import Game
from util import db
from classes.Agent import Agent
from classes.Epsilon import Epsilon
import logging
import util.util as util
from classes.Human import Human

logging.basicConfig(format='%(levelname)s:%(message)s', level=cfg.logging_level)


def benchmark_test(model, benchmark_model, benchmark_bot_name, run_id=None):
    winner_list = []
    epsilon = Epsilon(epsilon_func='eval')
    player_1 = Agent(name=cfg.bot_1_name, model=model, epsilon=epsilon)
    player_2 = Agent(name=benchmark_bot_name, model=benchmark_model, epsilon=epsilon)

    player_1.model.policy_net.eval()

    for j in range(cfg.random_bot_cycles):
        player_list = [player_1, player_2]
        game = Game(name="pinochle", players=player_list, run_id=run_id, current_cycle=None)
        game.deal()
        winner_list.append(game.play())

    return 1 - sum(winner_list) / len(winner_list)


def human_test(model):
    epsilon = Epsilon(epsilon_func='eval')
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


def get_average_reward(run_id, previous_experience_id, agent_id):
    df = db.get_rewards_by_id(run_id, previous_experience_id, agent_id)
    logging.debug(cfg.benchmark_freq * cfg.episodes_per_cycle)
    logging.debug(df.sum())
    average = df.sum() / (cfg.benchmark_freq * cfg.episodes_per_cycle)

    logging.info("Average reward since last benchmark: " + str(round(average.reward, 2)))

    return average.reward
