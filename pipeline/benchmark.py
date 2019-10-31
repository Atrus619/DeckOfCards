from config import Config as cfg
from pinochle.scripted_bots.RandomBot import RandomBot
from pinochle.Game import Game
from util import db
from classes.Agent import Agent
import logging
import util.util as util
from classes.Human import Human

logging.basicConfig(format='%(levelname)s:%(message)s', level=cfg.logging_level)


def random_bot_test(agent):
    winner_list = []
    player_2 = Agent(name=cfg.random_bot_name, model=RandomBot(), epsilon_func=util.get_random_bot_epsilon)
    agent.model.policy_net.eval()

    for j in range(cfg.random_bot_cycles):
        # Initialize game
        player_list = [agent, player_2]
        game = Game(name="pinochle", players=player_list, run_id=None, current_cycle=None)
        game.deal()
        winner_list.append(game.play())

    return 1 - sum(winner_list) / len(winner_list)


def human_test(model):
    player_2 = Human("Hades")
    model.policy_net.eval()

    # Set logging level to debug
    logging.getLogger().setLevel(logging.DEBUG)
    logging.info("Human test enabled, initializing AI uprising...")

    # Initialize game
    player_list = [model.player, player_2]
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

    logging.info("Agent: " + agent_id + "\tAverage reward: " + str(round(average.reward, 2)))

    return average.reward
