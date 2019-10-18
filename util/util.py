from classes.Agent import Agent
import logging
from config import Config as cfg
import util.db as db

logging.basicConfig(format='%(levelname)s:%(message)s', level=cfg.logging_level)


def list_to_dict(card_list):
    card_dict = {}

    for card in card_list:

        if card not in card_dict:
            card_dict[card] = 1
        else:
            card_dict[card] += 1

    return card_dict


def print_divider():
    logging.debug('---------------------------------------------------------------------------------')


def init_players(model_1, name_1, model_2, name_2, epsilon_func):
    # Returns a player_list to be used in setting up games based on passed Agents and models
    player_1 = Agent(name=name_1, model=model_1, epsilon_func=epsilon_func)
    player_2 = Agent(name=name_2, model=model_2, epsilon_func=epsilon_func)
    model_1.assign_player(player_1)
    model_2.assign_player(player_2)
    return [player_1, player_2]


def generate_run_id():
    if cfg.run_id == 'TEST':
        return cfg.run_id + '_' + str(db.get_global_max_id())
    else:
        return cfg.run_id


def get_epsilon_linear_anneal(current_cycle):
    """
    Returns an epsilon (probability of taking random action) based on the current cycle using linear annealing
    :param current_cycle: Current cycle of training
    """
    max_epsilon = cfg.max_epsilon
    min_epsilon = cfg.min_epsilon
    num_cycles = cfg.num_cycles

    return max(min_epsilon, max_epsilon - current_cycle / num_cycles * (max_epsilon - min_epsilon))


def get_epsilon_constant_decrement(current_cycle, decrement=None):
    """
    Returns an epsilon (probability of taking random action) based on the current cycle using a constant decrement
    Cannot go below the minimum epsilon in config
    :param current_cycle: Current training cycle
    :param decrement: Amount to decrease epsilon by per cycle
    """
    max_epsilon = cfg.max_epsilon
    min_epsilon = cfg.min_epsilon
    decrement = cfg.epsilon_decrement if decrement is None else decrement

    return max(min_epsilon, max_epsilon - (current_cycle - 1) * decrement)
