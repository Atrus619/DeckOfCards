from pinochle.Game import Game
from util import db
from torch.utils.data import DataLoader
from datasets.GameHistory import GameHistory
import pandas as pd


def play_game(name, players, run_id, current_cycle, config):
    """Single function to play a game to be used in async parallel processing"""
    game = Game(name=name, players=players, run_id=run_id, current_cycle=current_cycle, config=config)
    game.deal()
    return game.play()


def parse_game_output(game_output):
    # Splits stacked output (list of tuples of winrate, exp df) from game.play() into two distinct objects.
    winners, dfs = zip(*game_output)
    return list(winners), pd.concat(dfs) if any(x is not None for x in dfs) else None


def play_games(num_games, name, players, run_id, current_cycle, config):
    game_output = []
    for i in range(num_games):
        game_output.append(play_game(name=name, players=players, run_id=run_id, current_cycle=current_cycle, config=config))

    winner_list, df = parse_game_output(game_output=game_output)
    db.upload_exp(df=df)

    return winner_list


def train_model(model, config):
    # Import data from database based on experience replay buffer
    df = db.get_exp(run_id=config.run_id, buffer=config.experience_replay_buffer)
    gh = GameHistory(df=df, **config.GH_params)
    gh_gen = DataLoader(dataset=gh, batch_size=gh.batch_size, shuffle=True, num_workers=config.num_workers)

    # Train model
    model.train_self(num_epochs=config.epochs_per_cycle, exp_gen=gh_gen, is_storing_history=config.is_storing_history)
